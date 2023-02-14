import torch
import torch.nn as nn
from transformers import AutoModel
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class DimensionAttention(nn.Module):

    def __init__(self, word_embeddings):
        """
        :param word_embeddings: size=(vocab_size, 768), dtype=torch.FloatTensor
        """
        super(DimensionAttention, self).__init__()
        self.linear = nn.Linear(768, 768)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.embedding = nn.Embedding.from_pretrained(word_embeddings)

    def forward(self, word_index, dim_index, dim_attention_mask):
        """
        :param word_index: size=(batch_size, 300, 3), dtype=torch.LongTensor
        :param dim_index: size=(batch_size, 5), dtype=torch.LongTensor
        :param dim_attention_mask: size=(batch_size, 5), dtype=torch.LongTensor
        :return attention_dim_embedding: (batch_size, 5, 768)
        """
        batch_size = word_index.shape[0]
        num_dim = dim_index.shape[1]
        word_index = word_index[:, :, 0]  # size=(batch_size, 300)
        dim_attention_mask = dim_attention_mask.view(batch_size, -1, 1)  # (batch_size, 5, 1)
        word_embedding = self.embedding(word_index)  # size=(batch_size, 300, 768)
        dim_embedding = self.embedding(dim_index)  # size=(batch_size, 5, 768)
        padding_embedding = self.embedding(torch.LongTensor([0]).cuda())  # size=(1, 768)
        padding_embedding = padding_embedding.expand(batch_size, num_dim, 768)  # size=(batch_size, 5, 768)
        U = self.linear(word_embedding)  # size=(batch_size, 300, 768)
        U = self.tanh(U)  # size=(batch_size, 300, 768)
        alpha = self.softmax(
            torch.bmm(U, dim_embedding.transpose(1, 2))
        )  # size=(batch_size, 300, 5)
        attention_dim_embedding = torch.bmm(alpha.transpose(1, 2), word_embedding)  # size=(batch_size, 5, 768)
        attention_dim_embedding = attention_dim_embedding * (1 - dim_attention_mask) + \
            padding_embedding * dim_attention_mask  # size=(batch_size, 5, 768)
        return attention_dim_embedding


class TopKPool(nn.Module):

    def __init__(self, k):
        super(TopKPool, self).__init__()
        self.k = k

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, embedding_size)
        :return: (batch_size, seq_len)
        """
        output = x.topk(self.k, dim=-1)[0].mean(dim=-1)
        return output


class MainModule(nn.Module):

    def __init__(self, embeddings, pool_prop):
        """
        :param embeddings: size=(vocab_size, 768), dtype=numpy.array
        """
        super(MainModule, self).__init__()

        self.word_embeddings = torch.FloatTensor(embeddings).cuda()
        self.embedding = nn.Embedding.from_pretrained(self.word_embeddings, freeze=True)
        self.ernie = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        
        unfrozen_layer = ["layer.11", "pooler"]
        for param in self.ernie.named_parameters():
            if not any([kw in param[0] for kw in unfrozen_layer]):
                param[1].requires_grad = False

        self.dim_attention = DimensionAttention(self.word_embeddings)

        self.lstm_hid_dim = embeddings.shape[1] // 2  # must be <int> type
        self.bi_lstm = nn.LSTM(
            input_size=embeddings.shape[1], hidden_size=self.lstm_hid_dim,
            bidirectional=True, batch_first=True
        )

        # self.pool = nn.MaxPool1d(embeddings.shape[1], stride=1)
        self.topk_pool = TopKPool(int(self.lstm_hid_dim*2 * pool_prop))
        # self.linear = nn.Linear(517, 72)
        self.linear = nn.ModuleList([nn.Linear(517, 4) for i in range(18)])
        self.softmax = nn.Softmax(dim=-1)

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.lstm_hid_dim).cuda(),
                torch.randn(2, batch_size, self.lstm_hid_dim).cuda())

    def forward(self, input_ids, token_type_ids, attention_mask,
                word_index, dim_index, dim_attention_mask):
        """
        :param input_ids / token_type_ids / attention_mask: size=(batch_size, 512)
        :param word_index: size=(batch_size, 300, 3)
        :param dim_index: size=(batch_size, 5)
        :param dim_attention_mask: size=(batch_size, 5)
        :return: size=(batch_size, 18, 4)
        """
        batch_size = input_ids.shape[0]

        # get ERNIE char embeddings, size = (batch_size, 512, 768)
        ernie_output = self.ernie(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        ernie_char_embeddings = ernie_output.last_hidden_state

        # get dimension word embedding based on review-dimension attention mechanism
        # size = (batch_size, 5, 768)
        att_dim_embeddings = self.dim_attention(
            word_index, dim_index, dim_attention_mask
        )

        # fuse the ernie output and dimension word embeddings
        # if concatenate, size = (batch_size, 517, 768)
        fuse_embeddings = torch.concat([ernie_char_embeddings, att_dim_embeddings], dim=1)

        # Bi-LSTM layer, size = (batch_size, 517, 768)
        hidden_state = self.init_hidden(batch_size)
        lstm_output, hidden_state = self.bi_lstm(fuse_embeddings, hidden_state)
        
        # top-k pooling layer, size = (batch_size, 517)
        squeeze_output = self.topk_pool(lstm_output)

        output = torch.zeros(batch_size, 18, 4, device=device)
        for i in range(18):
            linear_output = self.linear[i](squeeze_output)
            output[:, i, :] = self.softmax(linear_output)

        return output
