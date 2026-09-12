# ============================================================
# PART 1: ĐỌC, LỌC, KIỂM TRA VÀ LÀM SẠCH DỮ LIỆU
# PHÂN TÍCH GIÁ BẤT ĐỘNG SẢN HÀ NỘI
# 6 THÁNG ĐẦU NĂM 2025
# ============================================================

import pandas as pd
import numpy as np

from config import DATA_FILE, CLEANED_DATA_FILE

# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================

def load_data():

    # Đọc dữ liệu bất động sản từ file CSV
    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("1. ĐỌC DỮ LIỆU")
    print("=" * 70)

    # Hiển thị kích thước dữ liệu ban đầu
    print("Kích thước dữ liệu ban đầu:", df.shape)

    # Hiển thị 5 dòng đầu tiên
    print("\n5 dòng đầu tiên:")
    print(df.head())

    return df


# ============================================================
# 2. KIỂM TRA DỮ LIỆU BAN ĐẦU
# ============================================================

def check_data(df):

    print("\n" + "=" * 70)
    print("2. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print("=" * 70)

    # Danh sách các cột
    print("\nDANH SÁCH CÁC CỘT:")
    print(df.columns.tolist())

    # Kiểu dữ liệu
    print("\nKIỂU DỮ LIỆU:")
    print(df.dtypes)

    # Số lượng dữ liệu thiếu
    print("\nSỐ LƯỢNG DỮ LIỆU THIẾU:")
    print(df.isnull().sum())

    # Tính tỷ lệ dữ liệu thiếu
    missing_percent = (
        df.isnull().sum() / len(df) * 100
    ).round(2)

    print("\nTỶ LỆ DỮ LIỆU THIẾU (%):")
    print(missing_percent)

    # Kiểm tra dữ liệu trùng
    print("\nSỐ DÒNG TRÙNG:")
    print(df.duplicated().sum())

    # Thống kê mô tả
    print("\nTHỐNG KÊ MÔ TẢ:")
    print(df.describe(include="all").transpose())


# ============================================================
# 3. LỌC DỮ LIỆU
# ============================================================

def filter_data(df):

    print("\n" + "=" * 70)
    print("3. LỌC DỮ LIỆU")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # 3.1. Chuyển published_at sang datetime
    # --------------------------------------------------------

    # Chuyển ngày đăng tin sang kiểu datetime
    # để có thể lọc theo thời gian
    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 3.2. Lọc bất động sản tại Hà Nội
    # --------------------------------------------------------

    df = df[
        df["province_name"].astype(str).str.strip() == "Hà Nội"
    ]

    print("\nSau khi lọc Hà Nội:")
    print("Số dòng:", len(df))

    # --------------------------------------------------------
    # 3.3. Lọc 6 tháng đầu năm 2025
    # --------------------------------------------------------

    # Khoảng thời gian:
    # Từ 01/01/2025 đến trước 01/07/2025
    df = df[
        (df["published_at"] >= "2025-01-01") &
        (df["published_at"] < "2025-07-01")
    ]

    print("\nSau khi lọc 6 tháng đầu năm 2025:")
    print("Số dòng:", len(df))

    # --------------------------------------------------------
    # 3.4. Chuyển price và area sang dạng số
    # --------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df["area"] = pd.to_numeric(
        df["area"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 3.5. Loại bỏ giá và diện tích không hợp lệ
    # --------------------------------------------------------

    # Giá và diện tích phải lớn hơn 0
    df = df[
        (df["price"] > 0) &
        (df["area"] > 0)
    ]

    # --------------------------------------------------------
    # 3.6. Loại bỏ dòng thiếu thông tin quan trọng
    # --------------------------------------------------------

    # Chỉ loại bỏ những dòng thiếu các thông tin
    # bắt buộc cho bài toán phân tích giá
    df = df.dropna(
        subset=[
            "price",
            "area",
            "district_name",
            "property_type_name",
            "published_at"
        ]
    )

    print("\nSau khi loại bỏ dữ liệu quan trọng không hợp lệ:")
    print("Số dòng:", len(df))

    return df


# ============================================================
# 4. XÓA DỮ LIỆU TRÙNG
# ============================================================

def remove_duplicates(df):

    print("\n" + "=" * 70)
    print("4. XỬ LÝ DỮ LIỆU TRÙNG")
    print("=" * 70)

    before = len(df)

    # Xóa các dòng giống hoàn toàn nhau
    df = df.drop_duplicates()

    after = len(df)

    print("Số dòng trùng đã xóa:", before - after)
    print("Số dòng còn lại:", after)

    return df


# ============================================================
# 5. XỬ LÝ CÁC CỘT CÓ NHIỀU DỮ LIỆU THIẾU
# ============================================================

def remove_high_missing_columns(df):

    print("\n" + "=" * 70)
    print("5. XỬ LÝ CÁC CỘT CÓ NHIỀU DỮ LIỆU THIẾU")
    print("=" * 70)

    # --------------------------------------------------------
    # Các cột có tỷ lệ thiếu dữ liệu quá cao sẽ được loại bỏ.
    #
    # Những cột dưới đây có tỷ lệ thiếu rất lớn:
    #
    # project_name       : 73,40%
    # floor_count        : 82,30%
    # house_depth        : 98,67%
    # road_width         : 92,83%
    # house_direction    : 73,50%
    # balcony_direction  : 81,53%
    #

    columns_to_drop = [
        "project_name",
        "floor_count",
        "house_depth",
        "road_width",
        "house_direction",
        "balcony_direction"
    ]

    # Chỉ xóa những cột thực sự tồn tại
    columns_to_drop = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    # Xóa các cột thiếu quá nhiều dữ liệu
    df = df.drop(
        columns=columns_to_drop
    )

    print("\nCác cột đã loại bỏ:")
    for col in columns_to_drop:
        print("-", col)

    # --------------------------------------------------------
    # QUAN TRỌNG:
    # Các cột sau được GIỮ LẠI dù có dữ liệu thiếu:
    #
    # ward_name        : 11,30%
    # street_name      : 32,73%
    # bedroom_count    : 44,87%
    # bathroom_count   : 48,47%
    # frontage_width   : 50,43%
    #
    # Các cột này sẽ được xử lý ở bước tiếp theo.
    # --------------------------------------------------------

    print("\nCác cột được giữ lại để xử lý dữ liệu thiếu:")
    keep_columns = [
        "ward_name",
        "street_name",
        "bedroom_count",
        "bathroom_count",
        "frontage_width"
    ]

    for col in keep_columns:
        if col in df.columns:
            print("-", col)

    return df


# ============================================================
# 6. XỬ LÝ DỮ LIỆU THIẾU
# ============================================================

def handle_missing_data(df):

    print("\n" + "=" * 70)
    print("6. XỬ LÝ DỮ LIỆU THIẾU")
    print("=" * 70)

    df = df.copy()

    # ========================================================
    # 6.1. CHUYỂN CÁC BIẾN SỐ SANG DẠNG SỐ
    # ========================================================

    numeric_columns = [
        "frontage_width",
        "bedroom_count",
        "bathroom_count"
    ]

    for col in numeric_columns:

        if col in df.columns:

            # Chuyển dữ liệu sang dạng số
            # Nếu không chuyển được thì thành NaN
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ========================================================
    # 6.2. GIỮ VÀ XỬ LÝ frontage_width
    # ========================================================

    if "frontage_width" in df.columns:

        # frontage_width có 50,43% dữ liệu thiếu
        # nhưng vẫn giữ lại vì đây là thông tin
        # có thể có ý nghĩa đối với giá bất động sản.

        median_value = df["frontage_width"].median()

        # Điền dữ liệu thiếu bằng giá trị trung vị
        df["frontage_width"] = df[
            "frontage_width"
        ].fillna(median_value)

        print(
            "frontage_width: đã giữ lại và điền thiếu bằng median =",
            median_value
        )

    # ========================================================
    # 6.3. GIỮ VÀ XỬ LÝ bedroom_count
    # ========================================================

    if "bedroom_count" in df.columns:

        # bedroom_count có 44,87% dữ liệu thiếu
        # nên giữ lại và xử lý dữ liệu thiếu.

        median_value = df["bedroom_count"].median()

        # Điền dữ liệu thiếu bằng trung vị
        df["bedroom_count"] = df[
            "bedroom_count"
        ].fillna(median_value)

        print(
            "bedroom_count: đã giữ lại và điền thiếu bằng median =",
            median_value
        )

    # ========================================================
    # 6.4. GIỮ VÀ XỬ LÝ bathroom_count
    # ========================================================

    if "bathroom_count" in df.columns:

        # bathroom_count có 48,47% dữ liệu thiếu
        # nên giữ lại và xử lý dữ liệu thiếu.

        median_value = df["bathroom_count"].median()

        # Điền dữ liệu thiếu bằng trung vị
        df["bathroom_count"] = df[
            "bathroom_count"
        ].fillna(median_value)

        print(
            "bathroom_count: đã giữ lại và điền thiếu bằng median =",
            median_value
        )

    # ========================================================
    # 6.5. GIỮ VÀ XỬ LÝ ward_name
    # ========================================================

    if "ward_name" in df.columns:

        # ward_name có 11,30% dữ liệu thiếu.
        # Đây là dữ liệu dạng chữ nên không dùng median.
        # Thay dữ liệu thiếu bằng "Không rõ".

        df["ward_name"] = df[
            "ward_name"
        ].fillna("Không rõ")

        print(
            "ward_name: đã giữ lại và thay dữ liệu thiếu bằng 'Không rõ'"
        )

    # ========================================================
    # 6.6. GIỮ VÀ XỬ LÝ street_name
    # ========================================================

    if "street_name" in df.columns:

        # street_name có 32,73% dữ liệu thiếu.
        # Đây là dữ liệu dạng chữ nên thay bằng "Không rõ".

        df["street_name"] = df[
            "street_name"
        ].fillna("Không rõ")

        print(
            "street_name: đã giữ lại và thay dữ liệu thiếu bằng 'Không rõ'"
        )

    return df


# ============================================================
# 7. TẠO CÁC BIẾN GIÁ
# ============================================================

def create_price_features(df):

    print("\n" + "=" * 70)
    print("7. TẠO CÁC BIẾN GIÁ")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # 7.1. Giá gốc
    # --------------------------------------------------------

    # Giá trong dataset gốc đã là VND
    df["price_original"] = df["price"]

    # --------------------------------------------------------
    # 7.2. Giá VND
    # --------------------------------------------------------

    # Không nhân với 1.000.000.000
    # vì price trong dữ liệu gốc đã là VND.
    df["price_vnd"] = df["price"]

    # --------------------------------------------------------
    # 7.3. Giá trên mỗi mét vuông
    # --------------------------------------------------------

    # Công thức:
    # Giá/m² = Giá bất động sản / Diện tích
    df["price_per_m2"] = (
        df["price_vnd"] / df["area"]
    )

    # Loại bỏ giá trị vô cực
    df["price_per_m2"] = df[
        "price_per_m2"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Những dòng không tính được giá/m²
    # sẽ được loại bỏ
    df = df.dropna(
        subset=["price_per_m2"]
    )

    print("Đã tạo:")
    print("- price_original")
    print("- price_vnd")
    print("- price_per_m2")

    return df


# ============================================================
# 8. XỬ LÝ THỜI GIAN
# ============================================================

def create_time_features(df):

    print("\n" + "=" * 70)
    print("8. XỬ LÝ DỮ LIỆU THỜI GIAN")
    print("=" * 70)

    df = df.copy()

    # Sắp xếp dữ liệu theo ngày đăng tin
    df = df.sort_values(
        by="published_at"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 8.1. Năm
    # --------------------------------------------------------

    df["year"] = df["published_at"].dt.year

    # --------------------------------------------------------
    # 8.2. Tháng
    # --------------------------------------------------------

    df["month"] = df["published_at"].dt.month

    # --------------------------------------------------------
    # 8.3. Quý
    # --------------------------------------------------------

    df["quarter"] = df["published_at"].dt.quarter

    # --------------------------------------------------------
    # 8.4. Thứ trong tuần
    # --------------------------------------------------------

    # Monday = 0
    # Sunday = 6
    df["day_of_week"] = (
        df["published_at"].dt.dayofweek
    )

    # --------------------------------------------------------
    # 8.5. Cuối tuần
    # --------------------------------------------------------

    # Thứ 7 và Chủ nhật = 1
    # Các ngày còn lại = 0
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    print("\nĐã tạo:")
    print("- year")
    print("- month")
    print("- quarter")
    print("- day_of_week")
    print("- is_weekend")

    # --------------------------------------------------------
    # Kiểm tra số lượng bản ghi theo tháng
    # --------------------------------------------------------

    print("\nSỐ LƯỢNG TIN ĐĂNG THEO THÁNG:")

    monthly_count = (
        df["month"]
        .value_counts()
        .sort_index()
    )

    print(monthly_count)

    return df


# ============================================================
# 9. SẮP XẾP LẠI CÁC CỘT
# ============================================================

def reorder_columns(df):

    print("\n" + "=" * 70)
    print("9. SẮP XẾP CÁC CỘT")
    print("=" * 70)

    preferred_columns = [

        # ----------------------------------------------------
        # Thông tin bất động sản
        # ----------------------------------------------------
        "name",
        "description",
        "property_type_name",

        # ----------------------------------------------------
        # Khu vực
        # ----------------------------------------------------
        "province_name",
        "district_name",
        "ward_name",
        "street_name",

        # ----------------------------------------------------
        # Giá và diện tích
        # ----------------------------------------------------
        "price_original",
        "price_vnd",
        "area",
        "price_per_m2",

        # ----------------------------------------------------
        # Thông tin căn nhà
        # ----------------------------------------------------
        "frontage_width",
        "bedroom_count",
        "bathroom_count",

        # ----------------------------------------------------
        # Thời gian
        # ----------------------------------------------------
        "published_at",
        "year",
        "month",
        "quarter",
        "day_of_week",
        "is_weekend"
    ]

    # Chỉ lấy các cột thực sự tồn tại
    existing_columns = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    # Các cột chưa nằm trong danh sách trên
    remaining_columns = [
        col
        for col in df.columns
        if col not in existing_columns
    ]

    # Sắp xếp lại
    df = df[
        existing_columns + remaining_columns
    ]

    print("\nThứ tự các cột sau khi sắp xếp:")
    print(df.columns.tolist())

    return df


# ============================================================
# 10. KIỂM TRA SAU KHI LÀM SẠCH
# ============================================================

def check_after_cleaning(df):

    print("\n" + "=" * 70)
    print("10. KIỂM TRA SAU KHI LÀM SẠCH")
    print("=" * 70)

    # Kích thước
    print("\nKÍCH THƯỚC DỮ LIỆU:")
    print(df.shape)

    # Số lượng thiếu
    print("\nSỐ LƯỢNG DỮ LIỆU THIẾU:")
    print(df.isnull().sum())

    # Tỷ lệ thiếu
    missing_percent = (
        df.isnull().sum() / len(df) * 100
    ).round(2)

    print("\nTỶ LỆ DỮ LIỆU THIẾU (%):")
    print(missing_percent)

    # Dòng trùng
    print("\nSỐ DÒNG TRÙNG:")
    print(df.duplicated().sum())

    # Thống kê giá
    print("\nTHỐNG KÊ GIÁ BẤT ĐỘNG SẢN - VND:")
    print(df["price_vnd"].describe())

    # Thống kê diện tích
    print("\nTHỐNG KÊ DIỆN TÍCH - m²:")
    print(df["area"].describe())

    # Thống kê giá/m²
    print("\nTHỐNG KÊ GIÁ/M² - VND/m²:")
    print(df["price_per_m2"].describe())

    # --------------------------------------------------------
    # Kiểm tra 5 trường được yêu cầu giữ lại
    # --------------------------------------------------------

    print("\nKIỂM TRA 5 TRƯỜNG ĐƯỢC GIỮ LẠI:")

    keep_columns = [
        "ward_name",
        "street_name",
        "bedroom_count",
        "bathroom_count",
        "frontage_width"
    ]

    for col in keep_columns:

        if col in df.columns:

            missing = df[col].isnull().sum()

            print(
                f"- {col}: còn {missing} giá trị thiếu"
            )

    # Hiển thị 10 dòng đầu
    print("\n10 DÒNG DỮ LIỆU SAU KHI LÀM SẠCH:")

    display_columns = [
        "name",
        "district_name",
        "ward_name",
        "street_name",
        "price_vnd",
        "area",
        "frontage_width",
        "bedroom_count",
        "bathroom_count",
        "price_per_m2",
        "published_at"
    ]

    # Chỉ lấy những cột thực sự tồn tại
    display_columns = [
        col
        for col in display_columns
        if col in df.columns
    ]

    display_data = df[
        display_columns
    ].head(10).copy()

    # Chỉ định dạng khi hiển thị
    if "price_vnd" in display_data.columns:

        display_data["price_vnd"] = (
            display_data["price_vnd"].map(
                lambda x: f"{x:,.0f}"
            )
        )

    if "area" in display_data.columns:

        display_data["area"] = (
            display_data["area"].map(
                lambda x: f"{x:,.2f}"
            )
        )

    if "frontage_width" in display_data.columns:

        display_data["frontage_width"] = (
            display_data["frontage_width"].map(
                lambda x: f"{x:,.2f}"
            )
        )

    if "price_per_m2" in display_data.columns:

        display_data["price_per_m2"] = (
            display_data["price_per_m2"].map(
                lambda x: f"{x:,.0f}"
            )
        )

    print(display_data)


# ============================================================
# 11. LƯU FILE
# ============================================================

def save_data(df):

    # Lưu thành MỘT file CSV duy nhất
    df.to_csv(
        CLEANED_DATA_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("11. LƯU DỮ LIỆU")
    print("=" * 70)

    print("\nĐã tạo file dữ liệu sạch:")
    print(CLEANED_DATA_FILE)


# ============================================================
# 12. CHƯƠNG TRÌNH CHÍNH
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Bước 1: Đọc dữ liệu
    # --------------------------------------------------------
    df = load_data()

    # --------------------------------------------------------
    # Bước 2: Kiểm tra dữ liệu ban đầu
    # --------------------------------------------------------
    check_data(df)

    # --------------------------------------------------------
    # Bước 3: Lọc Hà Nội + 6 tháng đầu năm 2025
    # --------------------------------------------------------
    df = filter_data(df)

    # --------------------------------------------------------
    # Bước 4: Xóa dữ liệu trùng
    # --------------------------------------------------------
    df = remove_duplicates(df)

    # --------------------------------------------------------
    # Bước 5: Loại bỏ các cột có quá nhiều dữ liệu thiếu
    # --------------------------------------------------------
    df = remove_high_missing_columns(df)

    # --------------------------------------------------------
    # Bước 6: Giữ và xử lý 5 trường có dữ liệu thiếu
    # --------------------------------------------------------
    df = handle_missing_data(df)

    # --------------------------------------------------------
    # Bước 7: Tạo các biến liên quan đến giá
    # --------------------------------------------------------
    df = create_price_features(df)

    # --------------------------------------------------------
    # Bước 8: Tạo các biến thời gian
    # --------------------------------------------------------
    df = create_time_features(df)

    # --------------------------------------------------------
    # Bước 9: Sắp xếp lại các cột
    # --------------------------------------------------------
    df = reorder_columns(df)

    # --------------------------------------------------------
    # Bước 10: Kiểm tra kết quả
    # --------------------------------------------------------
    check_after_cleaning(df)

    # --------------------------------------------------------
    # Bước 11: Lưu DUY NHẤT một file dữ liệu sạch
    # --------------------------------------------------------
    save_data(df)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH PART 1")
    print("=" * 70)