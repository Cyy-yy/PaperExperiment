import torch
import torch.nn.functional as F


class BasicModule(torch.nn.Module):
    """
    Define a basic module class
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        Load the saved model params
        """
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        """
        Save model
        """
        if path is None:
            raise ValueError('Please specify the saving path!!!')
        torch.save(self.state_dict(), path)
        return path


class TextCNN(BasicModule):
    """
    Construct a CNN model
    """
    def __init__(self, batch_size, embedding_size, kernel_size, kernel_num, num_classes):
        """
        :param kernel_size: <list> type
        :param kernel_num: <list> type, if sum(kernel_num)==768, then kernel_num=[256, 256, 256]
        """
        super(TextCNN, self).__init__()
        self.conv_lst = torch.nn.ModuleList([
            torch.nn.Conv1d(embedding_size, kn, kernel_size=ks, stride=1) for ks, kn in zip(kernel_size, kernel_num)
        ])
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        """
        :param x: size=[batch_size, seq_len, embedding_size]
        :return: size=[batch_size, 1, sum(kernel_num)]
        """
        x = x.permute(0, 2, 1)  # size=[batch_size, embedding_size, seq_len]
        output = [F.relu(conv(x)) for conv in self.conv_lst]  # size=[batch_size, 1, seq_len-kernel_size+1]
        output = [self.dropout(o) for o in output]
        output = [torch.nn.MaxPool1d(o.size(2))(o).permute(0, 2, 1) for o in output]  # size=[batch_size, 1, kernel_num]
        output = torch.cat(output, dim=2)  # size=[batch_size, 1, sum(kernel_num)]
        return output


class StructuredSelfAttention(BasicModule):

    def __init__(self,
                 batch_size,
                 lstm_hid_dim,
                 d_a,
                 n_classes,
                 label_embed,
                 embeddings,
                 embedding_size,
                 kernel_size,
                 kernel_num):
        """
        :d_a: output dimension for the first linear layer
        :n_classes: number of classes
        :embeddings: pretrained word embeddings
        """
        super(StructuredSelfAttention, self).__init__()
        self.n_classes = n_classes  # number of classes
        self.embeddings = self._load_embeddings(embeddings)  # word embeddings of texts
        self.label_embed = self.load_labelembedd(label_embed)  # word embeddings of labels
        self.lstm = torch.nn.LSTM(
            embedding_size, hidden_size=lstm_hid_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.linear_first = torch.nn.Linear(lstm_hid_dim*2, d_a)  # linear layer to compute W_1H
        self.linear_second = torch.nn.Linear(d_a, n_classes)  # linear layer to compute W_2tanh(W_1H)

        self.weight1 = torch.nn.Linear(lstm_hid_dim*2, 1)  # linear layer to compute the weight of M^(s)
        self.weight2 = torch.nn.Linear(lstm_hid_dim*2, 1)  # linear layer to compute the weight of M^(l)

        self.output_layer = torch.nn.Linear(lstm_hid_dim*2, n_classes)  # linear layer for output, which follows a sigmoid function
        self.dropout = torch.nn.Dropout(p=0.3)  # dropout op
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim

        # self.layer_norm = torch.nn.LayerNorm(embedding_size)
        self.text_cnn = TextCNN(batch_size, embedding_size, kernel_size, kernel_num, n_classes)

    def _load_embeddings(self, embeddings):
        """
        Load the embeddings based on flag, maybe pretrained word embeddings
        """
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings
    
    def load_labelembedd(self, label_embed):
        """
        Load the embeddings based on flag, maybe pretrained word embeddings
        """
        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def init_hidden(self, batch_size):
        """
        initialize the h_0, c_0 for LSTM, change `.cuda()` to `.to(device)`
        """
        return (torch.randn(2, batch_size, self.lstm_hid_dim).cuda(),torch.randn(2, batch_size, self.lstm_hid_dim).cuda())

    def forward(self, x):
        embeddings = self.embeddings(x)  # (batch_size, seq_len, embedding_size)
        # embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)  # (batch_size, seq_len, embedding_size)
        
        batch_size = embeddings.size(0)
        
        #step1 get LSTM outputs
        hidden_state = self.init_hidden(batch_size)  # two tensors whose size=(2*1, batch_size, lstm_hid_dim) 
        outputs, hidden_state = self.lstm(embeddings, hidden_state)  # output_size=(batch_size, seq_len, 2*lstm_hid_dim)

        # add to step1:
        cnn_outputs = self.text_cnn(embeddings)  # (batch_size, 1, 2*lstm_hid_dim)
        
        #step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))  # (batch_size, seq_len, d_a)
        selfatt = self.linear_second(selfatt)  # (batch_size, seq_len, n_classes)
        selfatt = F.softmax(selfatt, dim=1)  # (batch_size, seq_len, n_classes)
        selfatt= selfatt.transpose(1, 2)  # (batch_size, n_classes, seq_len)
        self_att = torch.bmm(selfatt, outputs)  # got M^(s), size=(batch_size, n_classes, 2*lstm_hid_dim)
        
        #step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]  # unidirectional hidden output of bi_lstm, size=(batch_size, seq_len, lstm_hid_dim)
        h2 = outputs[:, :, self.lstm_hid_dim:]  # unidirectional hidden output of bi_lstm, size=(batch_size, seq_len, lstm_hid_dim)
        
        label = self.label_embed.weight.data  # label word embeddings, size=(n_classes, lstm_hid_dim)
        m1 = torch.bmm(label.expand(batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))  # CH_1=(batch_size, n_classes, seq_len)
        m2 = torch.bmm(label.expand(batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))  # CH_2=(batch_size, n_classes, seq_len)
        label_att= torch.cat((torch.bmm(m1,h1),torch.bmm(m2,h2)),2)  # got M^(l) (batch_size, n_classes, 2*lstm_hid_dim)
        # label_att = F.normalize(label_att, p=2, dim=-1)
        # self_att = F.normalize(self_att, p=2, dim=-1) #all can
        weight1 = torch.sigmoid(self.weight1(label_att))  # (batch_size, n_classes, 1)
        weight2 = torch.sigmoid(self.weight2(self_att))  # (batch_size, n_classes, 1)
        weight1 = weight1 / (weight1 + weight2)  # scale to [0, 1]
        weight2 = 1 - weight1  # scale to [0, 1]

        doc = weight1*label_att + weight2*self_att  # (batch_size, n_classes, 2*lstm_hid_dim)
        # there two method, for simple, just add
        # also can use linear to do it
        avg_sentence_embeddings = torch.sum(doc, 1) / self.n_classes  # (batch_size, 2*lstm_hid_dim)
        # sentence_embeddings = torch.cat([avg_sentence_embeddings, cnn_outputs.squeeze(1)], dim=1)  # (batch_size, 4*lstm_hid_dim)
        cnn_embeddings = cnn_outputs.squeeze(1)  # (batch_size, 2*lstm_hid_dim)
        sentence_embeddings = avg_sentence_embeddings + cnn_embeddings
        sentence_embeddings = self.dropout(sentence_embeddings)
        pred = torch.sigmoid(self.output_layer(sentence_embeddings))  # (batch_size, n_classes)
        
        return pred
        
