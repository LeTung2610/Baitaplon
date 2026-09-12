# ============================================================
# PART 3: TRỰC QUAN HÓA DỮ LIỆU
# Biểu đồ 1: Giá bất động sản trung bình theo quận/huyện
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from config import CLEANED_DATA_FILE


# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================

df = pd.read_csv(CLEANED_DATA_FILE)


# ============================================================
# 2. TÍNH GIÁ BẤT ĐỘNG SẢN TRUNG BÌNH THEO QUẬN/HUYỆN
# ============================================================

district_price = (
    df.groupby("district_name")["price_vnd"]
    .mean()
    .sort_values(ascending=False)
)


# ============================================================
# 3. VẼ BIỂU ĐỒ
# ============================================================

plt.figure(figsize=(12, 7))

bars = plt.bar(
    district_price.index,
    district_price / 1_000_000_000
)


plt.title(
    "Giá bất động sản trung bình theo quận/huyện tại Hà Nội"
)

plt.xlabel("Quận/Huyện")

plt.ylabel("Giá trung bình (tỷ đồng)")


plt.xticks(
    rotation=45,
    ha="right"
)


# ============================================================
# 4. HIỂN THỊ GIÁ TRỊ TRÊN ĐẦU MỖI CỘT
# ============================================================

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )


plt.tight_layout()

plt.show()

# ============================================================
# 4. BIỂU ĐỒ 2: PHÂN BỐ GIÁ BẤT ĐỘNG SẢN
# ============================================================

# Chỉ lấy dữ liệu từ 0 đến 100 tỷ để dễ quan sát
# Không xóa dữ liệu gốc trong DataFrame
price_histogram = df.loc[
    df["price_vnd"] <= 100_000_000_000,
    "price_vnd"
] / 1_000_000_000


plt.figure(figsize=(12, 7))


# ============================================================
# 5. VẼ HISTOGRAM
# ============================================================

# 20 khoảng trong phạm vi 0 - 100 tỷ
# Mỗi khoảng tương ứng 5 tỷ đồng
counts, bins, patches = plt.hist(
    price_histogram,
    bins=20
)


plt.title(
    "Phân bố giá bất động sản tại Hà Nội (0 - 100 tỷ đồng)"
)

plt.xlabel("Giá bất động sản (tỷ đồng)")

plt.ylabel("Số lượng bất động sản")


# ============================================================
# 6. HIỂN THỊ SỐ LƯỢNG TRÊN ĐẦU CỘT
# ============================================================

for count, patch in zip(counts, patches):

    if count > 0:

        plt.text(
            patch.get_x() + patch.get_width() / 2,
            count,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=8
        )


# ============================================================
# 7. ĐIỀU CHỈNH TRỤC X
# ============================================================

# Hiển thị các mốc giá cách nhau 5 tỷ
plt.xticks(
    range(0, 101, 5),
    rotation=45
)


plt.xlim(0, 100)

plt.tight_layout()

plt.show()