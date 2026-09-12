# ============================================================
# PART 2: PHÂN TÍCH VÀ KHẢO SÁT DỮ LIỆU (EDA)
# Đề tài: Phân tích các yếu tố ảnh hưởng đến giá BĐS tại Hà Nội
# Thời gian: 6 tháng đầu năm 2025
# ============================================================

import pandas as pd
import numpy as np

from config import CLEANED_DATA_FILE


# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("1. ĐỌC DỮ LIỆU")
print("=" * 70)

df = pd.read_csv(CLEANED_DATA_FILE)

print(f"\nSố dòng: {len(df):,}")
print(f"Số cột: {len(df.columns)}")

print("\n5 dòng đầu tiên:")

print(
    df.head().to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)
# ============================================================
# 2. KIỂM TRA TỔNG QUAN
# ============================================================

print("\n" + "=" * 70)
print("2. KIỂM TRA TỔNG QUAN")
print("=" * 70)

print("\nKiểu dữ liệu:")
print(df.dtypes)

print("\nKích thước dữ liệu:")
print(df.shape)


# ============================================================
# 3. KIỂM TRA DỮ LIỆU THIẾU VÀ TRÙNG
# ============================================================

print("\n" + "=" * 70)
print("3. KIỂM TRA DỮ LIỆU THIẾU VÀ TRÙNG")
print("=" * 70)

missing = pd.DataFrame({
    "so_luong_thieu": df.isnull().sum(),
    "ty_le_thieu_%": (df.isnull().sum() / len(df) * 100).round(2)
})

print("\nDữ liệu thiếu:")
print(missing[missing["so_luong_thieu"] > 0])

duplicate_count = df.duplicated().sum()

print(f"\nSố dòng trùng lặp: {duplicate_count}")


# ============================================================
# 4. THỐNG KÊ CÁC BIẾN SỐ
# ============================================================

print("\n" + "=" * 70)
print("4. THỐNG KÊ CÁC BIẾN SỐ")
print("=" * 70)

numeric_columns = [
    "price_vnd",
    "price_per_m2",
    "area",
    "frontage_width",
    "bedroom_count",
    "bathroom_count"
]

print("\nThống kê mô tả:")

# Chỉ định dạng hiển thị số, không thay đổi dữ liệu gốc
print(
    df[numeric_columns]
    .describe()
    .to_string(float_format=lambda x: f"{x:,.2f}")
)


# ============================================================
# 5. PHÂN TÍCH GIÁ BẤT ĐỘNG SẢN
# ============================================================

print("\n" + "=" * 70)
print("5. PHÂN TÍCH GIÁ BẤT ĐỘNG SẢN")
print("=" * 70)

# ------------------------------------------------------------
# Cách 1: Giá gốc VND
# ------------------------------------------------------------

print("\n--- Cách 1: Giá theo VND ---")

print(f"Giá thấp nhất : {df['price_vnd'].min():,.0f} VND")
print(f"Giá cao nhất  : {df['price_vnd'].max():,.0f} VND")
print(f"Giá trung bình: {df['price_vnd'].mean():,.0f} VND")
print(f"Giá trung vị  : {df['price_vnd'].median():,.0f} VND")


# ------------------------------------------------------------
# Cách 2: Giá quy đổi sang tỷ đồng để dễ đọc
# ------------------------------------------------------------

print("\n--- Cách 2: Giá theo tỷ đồng ---")

print(f"Giá thấp nhất : {df['price_vnd'].min() / 1e9:.2f} tỷ")
print(f"Giá cao nhất  : {df['price_vnd'].max() / 1e9:.2f} tỷ")
print(f"Giá trung bình: {df['price_vnd'].mean() / 1e9:.2f} tỷ")
print(f"Giá trung vị  : {df['price_vnd'].median() / 1e9:.2f} tỷ")


# ============================================================
# 6. PHÂN TÍCH THEO QUẬN
# ============================================================

print("\n" + "=" * 70)
print("6. PHÂN TÍCH THEO QUẬN")
print("=" * 70)

district_analysis = (
    df.groupby("district_name")
    .agg(
        so_luong=("price_vnd", "count"),
        gia_trung_binh_vnd=("price_vnd", "mean"),
        gia_trung_vi_vnd=("price_vnd", "median"),
        gia_m2_trung_binh=("price_per_m2", "mean")
    )
    .sort_values("gia_trung_binh_vnd", ascending=False)
)

# Tạo thêm cột tỷ đồng để dễ đọc
district_analysis["gia_trung_binh_ty"] = (
    district_analysis["gia_trung_binh_vnd"] / 1e9
)

print("\nGiá bất động sản theo quận:")

print(
    district_analysis.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# 7. PHÂN TÍCH THEO LOẠI BẤT ĐỘNG SẢN
# ============================================================

print("\n" + "=" * 70)
print("7. PHÂN TÍCH THEO LOẠI BẤT ĐỘNG SẢN")
print("=" * 70)

property_analysis = (
    df.groupby("property_type_name")
    .agg(
        so_luong=("price_vnd", "count"),
        gia_trung_binh_vnd=("price_vnd", "mean"),
        gia_trung_vi_vnd=("price_vnd", "median"),
        gia_m2_trung_binh=("price_per_m2", "mean")
    )
    .sort_values("so_luong", ascending=False)
)

property_analysis["gia_trung_binh_ty"] = (
    property_analysis["gia_trung_binh_vnd"] / 1e9
)

print("\nPhân tích theo loại bất động sản:")

print(
    property_analysis.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# 8. PHÂN TÍCH THEO THỜI GIAN
# ============================================================

print("\n" + "=" * 70)
print("8. PHÂN TÍCH THEO THỜI GIAN")
print("=" * 70)

monthly_analysis = (
    df.groupby("month")
    .agg(
        so_luong=("price_vnd", "count"),
        gia_trung_binh_vnd=("price_vnd", "mean"),
        gia_trung_vi_vnd=("price_vnd", "median"),
        gia_m2_trung_binh=("price_per_m2", "mean")
    )
    .sort_index()
)

monthly_analysis["gia_trung_binh_ty"] = (
    monthly_analysis["gia_trung_binh_vnd"] / 1e9
)

print("\nPhân tích theo tháng:")

print(
    monthly_analysis.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# 9. PHÂN TÍCH TƯƠNG QUAN
# ============================================================

print("\n" + "=" * 70)
print("9. PHÂN TÍCH TƯƠNG QUAN")
print("=" * 70)

correlation_columns = [
    "price_vnd",
    "price_per_m2",
    "area",
    "frontage_width",
    "bedroom_count",
    "bathroom_count"
]

correlation_matrix = df[correlation_columns].corr()

print("\nMa trận tương quan:")

print(
    correlation_matrix.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nTương quan với giá bất động sản:")

price_correlation = (
    correlation_matrix["price_vnd"]
    .sort_values(ascending=False)
)

print(
    price_correlation.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 10. KIỂM TRA GIÁ TRỊ NGOẠI LAI
# ============================================================

print("\n" + "=" * 70)
print("10. KIỂM TRA GIÁ TRỊ NGOẠI LAI")
print("=" * 70)


def check_outliers(data, column):
    """
    Xác định số lượng ngoại lai bằng phương pháp IQR.
    """

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower) |
        (data[column] > upper)
    ]

    return len(outliers), lower, upper


outlier_columns = [
    "price_vnd",
    "price_per_m2",
    "area"
]

for column in outlier_columns:

    count, lower, upper = check_outliers(
        df,
        column
    )

    print(f"\n{column}")
    print(f"Số ngoại lai: {count}")
    print(f"Ngưỡng dưới: {lower:,.2f}")
    print(f"Ngưỡng trên: {upper:,.2f}")


# ============================================================
# KẾT THÚC PART 2
# ============================================================

print("\n" + "=" * 70)
print("HOÀN THÀNH PART 2 - EDA")
print("=" * 70)