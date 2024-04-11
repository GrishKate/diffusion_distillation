import torch
import torch.nn as nn


def sinusoidal_embedding(n, d):
    # n - размерность исходных данных (в нашем случае число моментов времени)
    # d - выходная размерность
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])  # коэффициенты для d гармоник
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1)).float()
    emb = t @ wk
    embedding = torch.cat((torch.sin(emb[:, :d // 2]), torch.cos(emb[:, d // 2:])), dim=1)
    # заполните половину из d компонент синунами sin(wk*t), оставшуюся косинусами cos(wk*t), где wk - коэффициенты гармоник
    return embedding


class ResBlock(nn.Module):
    def __init__(self, cin, cout, time_ch, n_gr=8, drop=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_gr, cin)
        self.act1 = nn.Sigmoid()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=(3, 3), padding=(1, 1))
        self.time_fc = nn.Linear(time_ch, cin)
        self.time_act = nn.Sigmoid()
        self.c = [cin, cout, time_ch]

    def forward(self, x, time):
        y = self.time_fc(time * self.time_act(time))[:, :, None, None]
        x += y
        out = self.norm1(x)
        out = self.conv1(out * self.act1(out))
        return out


class Attention(nn.Module):
    def __init__(self, cin, heads=1, d=None, n_gr=8):
        super().__init__()
        if d is None:
            d = cin
        self.norm = nn.GroupNorm(n_gr, cin)
        self.fc = nn.Linear(cin, heads * d * 3)
        self.last_fc = nn.Linear(heads * d, cin)
        self.scale = d ** -0.5
        self.heads = heads
        self.d = d

    def forward(self, x, t=None):
        bs, cin, h, w = x.shape
        x = x.reshape(bs, cin, -1).permute(0, 2, 1)
        qkv = self.fc(x).reshape(bs, -1, self.heads, 3 * self.d)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        out = torch.einsum('bijh,bjhd->bihd', attn, v).reshape(bs, -1, self.heads * self.d)
        out = self.last_fc(out) + x
        out = out.permute(0, 2, 1).reshape(bs, cin, h, w)
        return out


class UpBlock(nn.Module):
    def __init__(self, cin, cout, time_ch, has_attn):
        super().__init__()
        self.res_blk = ResBlock(cin + cout, cout, time_ch)
        if has_attn:
            self.attn = Attention(cout)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        return self.attn(self.res_blk(x, t))


class Middle(nn.Module):
    def __init__(self, cout, ch):
        super().__init__()
        self.res1 = ResBlock(cout, cout, ch)
        self.attn = Attention(cout)
        self.res2 = ResBlock(cout, cout, ch)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x, t)
        x = self.res2(x, t)
        return x


class DownBlock(nn.Module):
    def __init__(self, cin, cout, time_ch, has_attn):
        super().__init__()
        self.res = ResBlock(cin, cout, time_ch)
        if has_attn:
            self.attn = Attention(cout)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=16):
        super().__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim * 4)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim * 4)
        self.time_embed.requires_grad_(False)  # мы эмбединг слой уже инициализировали и менять его не будем

        cout, cin, ch = [time_emb_dim] * 3
        B = 1
        res = [1, 2, 2]
        has_attn = [False, False, True, True]
        self.conv1 = nn.Conv2d(3, ch, kernel_size=(3, 3), padding=(1, 1))
        down = []
        for i in range(len(res)):
            cout = cin * res[i]
            for _ in range(B):
                down.append(DownBlock(cin, cout, ch * 4, has_attn[i]))
                cin = cout
            if i < len(res) - 1:
                down.append(nn.Conv2d(cin, cin, (3, 3), (2, 2), (1, 1)))
        self.down = nn.ModuleList(down)
        self.middle = Middle(cout, ch * 4)
        up = []
        cin = cout
        for i in range(len(res) - 1, -1, -1):
            cout = cin
            for _ in range(B):
                up.append(UpBlock(cin, cout, ch * 4, has_attn[i]))
            cout = cin // res[i]
            up.append(UpBlock(cin, cout, ch * 4, has_attn[i]))
            cin = cout
            if i > 0:
                up.append(nn.ConvTranspose2d(cin, cin, (4, 4), (2, 2), (1, 1)))
        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.Sigmoid()
        self.last = nn.Conv2d(cin, 3, kernel_size=(3, 3), padding=(1, 1))

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x, time):
        t = self.time_embed(time)
        x = self.conv1(x)
        lst = [x]
        for layer in self.down:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
            else:
                x = layer(x, t)
            lst.append(x)
        x = self.middle(x, t)
        for layer in self.up:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
            else:
                connection = lst.pop()
                x = torch.cat((x, connection), dim=1)
                x = layer(x, t)
        x = self.norm(x)
        x = self.last(x * self.act(x))
        return x
