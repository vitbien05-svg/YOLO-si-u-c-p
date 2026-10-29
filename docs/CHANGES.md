# Các chỉnh sửa so với code gốc

Repo này dựa trên code TPH-YOLOv5 (nền YOLOv5 v6.0, năm 2021). Code gốc được viết cho các bản thư viện cũ. Môi trường hiện tại là Kaggle: Python 3.13, torch 2.11, numpy 2.x, Pillow ≥ 10, albumentations 2.x. Với các bản mới này, code gốc bị lỗi hoặc hết VRAM. File này ghi lại **mọi chỗ đã sửa trong code gốc**, kèm lý do. Các file mới thêm vào (yaml, sơ đồ) được liệt kê ở cuối.

Nguyên tắc chung: **chỉ sửa để code chạy được, không thay đổi kết quả tính toán của model**.

---

## 1. Sửa tương thích thư viện mới (commit `5765850`)

| File | Sửa | Lý do |
|---|---|---|
| `utils/datasets.py` (3 chỗ), `utils/general.py` (2 chỗ) | `.astype(np.int)` → `.astype(int)` | numpy ≥ 1.24 đã bỏ `np.int`, gặp là crash ngay lúc đọc dataset. `np.int` vốn chỉ là tên khác của `int`, nên kết quả không đổi. |
| `train.py`, `models/experimental.py`, `utils/general.py`, `utils/loggers/__init__.py` | `torch.load(...)` → `torch.load(..., weights_only=False)` | Từ torch 2.6, `torch.load` mặc định chỉ đọc tensor (`weights_only=True`). File `.pt` của YOLOv5 lưu cả object model nên bị từ chối. Thêm `weights_only=False` để đọc như trước. Chỉ dùng với file `.pt` do chính mình tạo ra, hoặc tải từ release chính thức của Ultralytics. |
| `utils/loss.py` (`build_targets`) | `gj.clamp_(0, gain[3] - 1)` → `gj.clamp_(0, int(gain[3]) - 1)` (tương tự với `gi`) | Từ torch 1.12, gọi `clamp_` trên tensor kiểu `long` với cận là số thực sẽ báo lỗi *"result type Float can't be cast to Long"*. `gain[3]` là kích thước lưới (số nguyên lưu dưới dạng float), nên ép về `int` không làm đổi giá trị. |
| `utils/plots.py` (`Annotator`) | `font.getsize(text)` → `font.getbbox(text)` | Pillow ≥ 10 đã xoá `FreeTypeFont.getsize`, nên vẽ ảnh `train_batch*.jpg` bị lỗi. `getbbox` trả về `(trái, trên, phải, dưới)`, chỉ lấy 2 số cuối làm rộng và cao như trước. Thay đổi này chỉ ảnh hưởng ảnh minh hoạ, không ảnh hưởng việc train. |

**Kiểm tra:** chạy thử train 1 epoch trên dữ liệu giả (CPU, torch 2.12, numpy 2.1) từ đầu đến cuối không lỗi. Pretrained `yolov5s.pt` (bản v6.0) nạp được 211/606 tensor vào model TPH-PFFN.

---

## 2. Attention của C3TR không lưu ma trận N × N (`models/common.py`, `TransformerLayer.forward`)

```python
# trước
x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x
# sau
x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_), need_weights=False)[0]) + x
```

### Vấn đề
Train `TPH-Yolov5-PFFN-neck.yaml` trên Kaggle (2× T4, mỗi con 14.5 GB) bị hết VRAM ngay ở batch đầu tiên. Lỗi xảy ra tại phép tính ma trận attention trong khối C3TR ở head P2:

| Cấu hình | Lỗi |
|---|---|
| imgsz 1280, batch 8 (4/GPU) | `CUDA out of memory. Tried to allocate 312.50 GiB` |
| imgsz 640, batch 8 (4/GPU) | `CUDA out of memory. Tried to allocate 19.53 GiB` |

Nguyên nhân: ở head P2 (stride 4), ảnh 640 cho feature map 160 × 160, tức **N = 25,600 token**. Attention tính điểm cho mọi cặp token, tạo ra ma trận N × N. Mặc định `nn.MultiheadAttention` dùng `need_weights=True`, nên **toàn bộ ma trận này được tạo ra và giữ trong VRAM**:

```
4 ảnh × 4 head × 25,600² × 2 byte (fp16) = 19.53 GiB      ← khớp đúng log
```

Giảm batch không giải quyết được: chỉ 1 ảnh đã cần khoảng 5 GiB cho forward, cộng thêm backward thì sát hoặc vượt 14.5 GB.

### Cách sửa
Với `need_weights=False`, torch (≥ 2.0) chuyển sang `scaled_dot_product_attention`. Kernel này tính attention theo từng khối nhỏ và **không bao giờ giữ nguyên ma trận N × N**. Bộ nhớ cho attention giảm từ tỉ lệ N² xuống tỉ lệ N.

Sửa vậy là an toàn vì:
- **Kết quả tính y hệt.** `need_weights` chỉ quyết định có trả bảng điểm attention ra ngoài hay không. Code luôn lấy `[0]` (output) và bỏ `[1]` (bảng điểm), nên không mất gì.
- **Kiến trúc không đổi**: số layer, số tham số và weight giữ nguyên. Checkpoint train bằng bản cũ vẫn dùng được với bản mới và ngược lại.
- Đã kiểm tra trên CPU (torch 2.12): output của 2 cách chênh tối đa **2.3e-8** ở cả chế độ `train()` và `eval()`. Đây là sai số làm tròn của số thực. Trong source của `F.multi_head_attention_forward` cũng xác nhận nhánh `need_weights=False` gọi `scaled_dot_product_attention`.

### Vì sao tác giả gốc không làm như vậy
- Paper TPH-YOLOv5 ra 08/2021 và dùng torch 1.x. Ở torch 1.x, `need_weights=False` **không tiết kiệm bộ nhớ**: ma trận vẫn được tính đầy đủ, chỉ không trả ra ngoài. Đường SDPA chỉ có từ **PyTorch 2.0 (03/2023)**.
- Khối `TransformerLayer` lấy từ YOLOv5 của Ultralytics. Ultralytics viết nó cho P5 (20 × 20 = 400 token), nơi ma trận N × N rất nhỏ. TPH đặt khối này xuống P2 (25,600 token), khiến bộ nhớ tăng khoảng 64² ≈ 4000 lần.
- Chính nhóm tác giả sau đó đã chuyển sang Swin (`C3STR`, attention theo cửa sổ 8 × 8) ở `models/yolov5l-xs-tph.yaml`, rồi sang CLLA ở TPH-YOLOv5++. Lý do là để tránh chi phí N².

### Lưu ý
- **Tốc độ tính không đổi.** Số phép tính của attention vẫn tỉ lệ N², chỉ có bộ nhớ là giảm. Vì vậy train ở 1280 vẫn chậm hơn 640 khoảng 16 lần ở các khối C3TR.
- Cần torch ≥ 2.0. Với torch 1.x, dòng này vẫn chạy nhưng không giảm được VRAM.

---

## Chưa sửa (đã biết)

| Chỗ | Vấn đề | Ảnh hưởng |
|---|---|---|
| `models/common.py:354` (`SwinTransformerLayer.create_mask`) | `(0, -self.window_size)` đáng lẽ là `slice(0, -self.window_size)` | Chỉ ảnh hưởng các yaml dùng `C3STR` (Swin). Các yaml đang train không dùng nên chưa sửa. |
| `utils/augmentations.py:32` | `ImageCompression(quality_lower=75)`: albumentations 2.x đã đổi tên tham số, nên báo `UserWarning` | Transform này đang tắt (`p=0.0`). Chỉ là cảnh báo, các augmentation còn lại vẫn chạy (log có dòng `albumentations: Blur, MedianBlur, ToGray, CLAHE, RandomBrightnessContrast`). |
| `utils/general.py`, `utils/loggers/` | Dùng `pkg_resources` (setuptools sẽ bỏ) | Trên Kaggle cài `setuptools<81` để giữ lại module này. |

---

## File thêm mới (không sửa code gốc)

| File | Nội dung |
|---|---|
| `models/yolov5s-tph-reimplement.yaml` | TPH-YOLOv5 theo Fig. 3 của paper, quy mô v5s: 4 head P2–P5, C3TR + CBAM |
| `models/yolov5s-pffn.yaml` | Neck PFFN v4 (Lu et al., Sensors 2023), yaml chính chủ của paper, dùng làm baseline |
| `models/TPH-Yolov5-PFFN-neck.yaml` | Ghép backbone/head của TPH với neck PFFN v4 |
| `diagrams/TPH-Yolov5-PFFN-neck.drawio` | Sơ đồ draw.io: page-2 vẽ từng layer, page-3 vẽ dạng gom khối như Fig. 4 |
| `diagrams/check_drawio.py` | Script đối chiếu mũi tên trong sơ đồ với cột `from` của yaml |
