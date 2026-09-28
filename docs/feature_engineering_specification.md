# Đặc tả Feature Engineering - Customer Churn Prediction

## 1. Mục đích của tài liệu

Tài liệu này giải thích đầy đủ notebook `notebooks/05_feature_engineering.ipynb` theo thứ tự thực thi. Mục tiêu là để người đọc hiểu:

- notebook nhận dữ liệu gì và tạo ra dữ liệu gì;
- từng nhóm feature được xử lý như thế nào;
- công thức của 9 feature mới;
- cách notebook ngăn target leakage và temporal leakage;
- các kiểm tra đã chạy và kết quả thực tế;
- feature nào được giữ để làm baseline, feature nào chỉ được thử nghiệm;
- phần nào vẫn thuộc notebook Preprocessing và Machine Learning.

Notebook này **chưa huấn luyện mô hình**. Nó chỉ chuẩn bị và kiểm tra feature.

## 2. Vị trí trong pipeline

```text
Data Understanding
  -> Data Cleaning
  -> EDA
  -> Statistical Analysis
  -> Feature Engineering (notebook này)
  -> Preprocessing
  -> Machine Learning
  -> Evaluation
  -> Explainability
```

Feature Engineering trả lời câu hỏi: "Từ các thuộc tính khách hàng đã clean, nên tạo và chọn những biến nào để chuyển sang bước preprocessing/ML?"

Nó không thực hiện:

- encoding categorical;
- scaling toàn bộ dữ liệu;
- imputation;
- chia chính thức train/validation/test cho ML;
- train Logistic Regression, Random Forest hoặc XGBoost;
- tính Accuracy, ROC-AUC, Precision, Recall hoặc F1;
- SHAP hoặc phân tích feature importance của mô hình.

## 3. Input và output

### 3.1 Input

Notebook đọc:

```text
data/processed/customer_churn_clean.csv
```

Kết quả chạy thực tế:

- 50.000 dòng;
- 25 cột;
- target: `Churned`;
- 24 predictor ban đầu: 20 numeric, 3 categorical được khai báo rõ ràng và 1 cột `City` được kiểm tra riêng.

`PROJECT_ROOT` được xác định từ thư mục notebook. Nếu thư mục hiện tại không chứa `data`, notebook thử thư mục cha.

### 3.2 Output

Notebook xuất:

```text
data/processed/customer_churn_feature_engineered.csv
```

Kết quả chạy thực tế:

- 50.000 dòng;
- 33 cột;
- gồm 32 feature được phép chuyển sang Preprocessing/ML và target `Churned`;
- không gồm `City`;
- không gồm các tên feature target-derived bị cấm.

## 4. Nguyên tắc chung

### 4.1 Không dùng target để tạo feature

`Churned` chỉ được giữ làm nhãn `y`. Không có feature nào được tính từ `Churned`, churn rate hoặc xác suất churn.

Ví dụ bị cấm:

```text
Churn_Risk_Score
Churn_Probability
Churn_Level
Churn_Status
```

### 4.2 Feature phải có ý nghĩa nghiệp vụ

Mỗi feature phải mô tả một đặc điểm, hành vi, giá trị hoặc tương tác có thể giải thích được của khách hàng.

### 4.3 Chỉ dùng thông tin có trước thời điểm dự đoán

Các biến phụ thuộc thời gian chỉ hợp lệ nếu dữ liệu là snapshot trước prediction time. Đặc biệt cần kiểm soát:

- `Days_Since_Last_Purchase`;
- `Total_Purchases`;
- `Lifetime_Value`;
- `Customer_Service_Calls`;
- `Credit_Balance`.

Nếu một trong các biến này chứa thông tin xảy ra sau thời điểm dự đoán, nó sẽ gây temporal leakage dù không chứa chữ `Churned`.

### 4.4 Tách Feature Engineering và Preprocessing

Notebook này tạo logic feature: ratios, nhóm, composite score và interaction. Notebook sau phải xử lý encoding, scaling, missing-value imputation và transform theo train data chính thức.

## 5. Trình tự thực thi của notebook

### Bước 1 - Import thư viện

Notebook import:

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
```

`seaborn.set_theme(style="whitegrid")` được dùng để các biểu đồ validation có cùng phong cách.

Output mong đợi:

```text
Libraries imported successfully.
```

### Bước 2 - Đọc và kiểm tra dataset

Notebook đọc CSV vào `df`, đặt:

```python
TARGET = "Churned"
```

Sau đó hiển thị path, shape, vài dòng đầu và kiểu dữ liệu. Nó lập `overview` gồm:

- số dòng;
- số cột;
- tổng số ô missing;
- số dòng duplicate.

Nó cũng tạo `target_distribution` gồm số lượng và phần trăm của từng giá trị `Churned`.

Hai assertion đầu tiên xác nhận:

1. `Churned` thật sự tồn tại;
2. các giá trị target khác missing chỉ thuộc `{0, 1}`.

Kết quả chạy thực tế xác nhận input có shape `(50000, 25)`.

### Bước 3 - Khai báo target và nhóm feature

Categorical được khai báo:

```text
Gender
Country
Signup_Quarter
```

Numerical được khai báo:

```text
Age
Membership_Years
Login_Frequency
Session_Duration_Avg
Pages_Per_Session
Cart_Abandonment_Rate
Wishlist_Items
Total_Purchases
Average_Order_Value
Days_Since_Last_Purchase
Discount_Usage_Rate
Returns_Rate
Email_Open_Rate
Customer_Service_Calls
Product_Reviews_Written
Social_Media_Engagement_Score
Mobile_App_Usage
Payment_Method_Diversity
Lifetime_Value
Credit_Balance
```

Notebook tạo:

```python
X_original = df.drop(columns=[TARGET])
y = df[TARGET].copy()
```

`X_original` là predictor ban đầu, còn `y` là target. `City` có trong input nhưng không nằm trong danh sách categorical baseline; nó được review riêng ở bước feature selection.

### Bước 4 - Kiểm tra leakage

Notebook lập bảng `leakage_df` để ghi lại risk của target và các biến phụ thuộc snapshot.

Nó kiểm tra tên mọi cột trong `X_original`: nếu tên cột chứa chuỗi `churn`, assertion sẽ fail. Nó cũng xác nhận `TARGET` không nằm trong `X_original`.

Kết quả chạy thực tế:

```text
Leakage gate passed: no target-derived predictor is present.
```

Điều này chỉ chứng minh không có target-derived predictor theo các kiểm tra hiện tại. Nó không tự động chứng minh timestamp của dữ liệu là đúng; phần snapshot phải được xác nhận từ data dictionary/business process.

## 6. Cách notebook fit tham số để giảm leakage

Notebook tạo một internal demonstration split bằng:

```python
RANDOM_STATE = 42
TRAIN_FRACTION = 0.80
```

Nó xáo trộn vị trí bằng `np.random.default_rng(42)`, lấy 80% dòng làm `fit_data` và dùng toàn bộ dữ liệu làm `transform_data`.

Kết quả chạy thực tế:

```text
Transformation fit rows: 40000
Transformation application rows: 50000
No target values were used to create the split or fit feature transformations.
```

Ý nghĩa:

- min/max của `Engagement_Score` chỉ học từ 40.000 dòng;
- quantile thresholds chỉ học từ 40.000 dòng;
- các tham số đó sau đó được áp dụng cho 50.000 dòng để minh họa transform.

Đây **chưa phải** train/validation/test split chính thức của ML. Khi làm ML, phải chia dữ liệu chính thức một lần, fit tất cả tham số feature/preprocessing trên `X_train`, rồi transform validation/test bằng tham số đã fit.

## 7. Chín feature được tạo

### 7.1 `Engagement_Score`

Mục đích: gom nhiều biểu hiện engagement về một điểm tổng hợp, tránh cộng trực tiếp các biến có scale khác nhau.

Input:

```text
Login_Frequency
Session_Duration_Avg
Pages_Per_Session
Email_Open_Rate
Mobile_App_Usage
Social_Media_Engagement_Score
```

Với mỗi input `x`, notebook tính min-max normalization bằng min/max lấy từ `fit_data`:

$$
normalized(x) = \frac{x - min_{fit}}{max_{fit} - min_{fit}}
$$

Nếu range bằng 0, range được thay bằng `1.0` để tránh chia cho 0. Sau đó:

$$
Engagement\_Score = mean(normalized\ inputs)
$$

Điểm quan trọng: notebook **không clip** giá trị về `[0, 1]`. Nếu dòng mới nằm ngoài train range, điểm có thể nhỏ hơn 0 hoặc lớn hơn 1. Đây là hành vi có chủ ý và cần được quyết định lại khi xây pipeline chính thức.

Decision: `TEST`.

### 7.2 `Purchase_Frequency`

Mục đích: biểu diễn purchase activity theo một đơn vị thời gian tenure.

Công thức:

$$
Purchase\_Frequency = \frac{Total\_Purchases}{Membership\_Years}
$$

Nếu `Membership_Years == 0`, denominator được thay bằng `NaN`, nhờ đó tránh chia cho 0 và tránh tạo `Inf`.

Tên gọi "frequency" chỉ hợp lệ nếu `Total_Purchases` thực sự là số đơn/số lần mua và `Membership_Years` là thời gian tương ứng. Nếu `Total_Purchases` là volume hoặc một đại lượng khác, cần đổi cách diễn giải.

Kết quả kiểm tra định nghĩa:

- `Total_Purchases`: min `0.00`, max `129.00`, không có giá trị thập phân;
- `Average_Order_Value`: min `26.38`, max `9666.379178`, có giá trị thập phân.

Notebook vẫn yêu cầu xác nhận data dictionary trước khi coi đây là purchases/year.

Decision: `TEST`.

### 7.3 `Purchase_Recency_Group`

Mục đích: biến số ngày từ lần mua gần nhất thành nhóm dễ diễn giải.

Input: `Days_Since_Last_Purchase`.

Ngưỡng được fit trên `fit_data` tại các quantile 25%, 50%, 75%. Bốn nhóm tạo ra là:

```text
Recent
Moderate
Inactive
Very_Inactive
```

Các bin là:

```text
(-inf, Q25]
(Q25, Q50]
(Q50, Q75]
(Q75, inf)
```

Decision: `TEST`.

### 7.4 `Purchase_Frequency_Group`

Mục đích: cung cấp biểu diễn categorical của `Purchase_Frequency` để model có thể học quan hệ phi tuyến theo nhóm.

Ngưỡng lấy từ quantile 33% và 67% của `Purchase_Frequency` trên phần fit. Nhóm:

```text
Low
Medium
High
```

Decision: `TEST`.

### 7.5 `Service_Call_Group`

Mục đích: so sánh cách dùng số cuộc gọi customer service dạng numeric với cách gom thành nhóm.

Input: `Customer_Service_Calls`.

Ngưỡng dùng Q25, Q50, Q75 trên `fit_data`. Nhóm:

```text
No_or_Few_Calls
Moderate
Frequent
Very_Frequent
```

Decision: `TEST`.

### 7.6 `Cart_Abandonment_Group`

Mục đích: kiểm tra quan hệ phi tuyến của cart abandonment bằng biểu diễn nhóm.

Input: `Cart_Abandonment_Rate`.

Ngưỡng dùng quantile 33% và 67% trên `fit_data`. Nhóm:

```text
Low
Medium
High
```

Decision: `TEST`.

### 7.7 `Estimated_Purchase_Value`

Mục đích: tạo một ước tính giá trị mua tích lũy.

Công thức:

$$
Estimated\_Purchase\_Value = Total\_Purchases \times Average\_Order\_Value
$$

Từ `Estimated` là bắt buộc vì đây không phải actual revenue đã được xác nhận. Công thức chỉ có ý nghĩa nếu:

- `Total_Purchases` là order count;
- `Average_Order_Value` là monetary value trung bình trên mỗi order;
- cả hai đều được tính trước thời điểm dự đoán.

Decision: `TEST`.

### 7.8 `Cart_Recency_Interaction`

Mục đích: biểu diễn đồng thời hai tín hiệu: bỏ giỏ hàng nhiều và lâu chưa mua.

Công thức:

$$
Cart\_Recency\_Interaction = Cart\_Abandonment\_Rate \times Days\_Since\_Last\_Purchase
$$

Giá trị cao gợi ý cả abandonment và recency đều cao, nhưng chỉ là feature ứng viên; chưa được kết luận là nguyên nhân churn.

Decision: `TEST`.

### 7.9 `Low_Engagement_Recency`

Mục đích: tạo cờ nhị phân cho nhóm khách hàng vừa engagement thấp vừa lâu chưa mua.

Ngưỡng:

- low engagement = Q25 của `Engagement_Score` trên phần fit;
- long recency = Q75 của `Days_Since_Last_Purchase` trên phần fit.

Logic:

```text
1 nếu Engagement_Score <= low_engagement_threshold
   và Days_Since_Last_Purchase >= long_recency_threshold
0 trong các trường hợp còn lại
```

Kết quả chạy thực tế:

- positive: 3.046 khách hàng;
- negative: 46.954 khách hàng;
- positive rate: `0.06092` tương đương `6.092%`.

Decision: `TEST`.

## 8. Feature dictionary và kiểm tra City

Notebook tạo `feature_dictionary` với các trường:

```text
Feature
Category
Source
Data_Used
Original_Engineered
Formula
Reason
Leakage_Risk
Data_Type
Validation
ML_Decision
Business_Meaning
```

Bảng này là metadata chính của feature, giúp biết mỗi cột đến từ đâu, tính thế nào và được phép dùng ra sao.

`City` được kiểm tra bằng crosstab `Country x City`:

- có 40 category;
- không có city nào xuất hiện ở nhiều country;
- decision thực tế: `EXCLUDE`.

Lý do: giữ baseline gọn và dành việc thử incremental predictive value của City cho experiment sau. `City` không bị loại vì target leakage.

## 9. Validation sau khi tạo feature

### 9.1 Missing, infinite, duplicate và constant

Notebook kiểm tra toàn bộ `df_fe`.

Kết quả thực tế:

| Kiểm tra | Count |
|---|---:|
| Missing values | 0 |
| Infinite numeric values | 0 |
| Duplicate columns | 0 |
| Duplicate rows | 0 |
| Constant numeric features | 0 |

Tất cả assertion đều PASS.

### 9.2 Near-zero variance và category frequency

Notebook ghi nhận các biến có unique ratio thấp để bước ML xem xét. Đây là cảnh báo phân phối, không phải tự động loại feature.

Một số kết quả đáng chú ý:

- `Low_Engagement_Recency`: unique ratio `0.00004`;
- `Payment_Method_Diversity`: `0.00010`;
- `Customer_Service_Calls`: `0.00042`;
- `City`: category nhỏ nhất chiếm `0.988%` và lớn nhất `7.098%`;
- `Gender`: nhỏ nhất `1.874%`, lớn nhất `50.232%`.

Điểm cần hiểu: một numeric feature có ít giá trị khác nhau không mặc nhiên vô dụng. Nó chỉ cần được theo dõi khi preprocessing/modeling.

### 9.3 Business range

Notebook kiểm tra các domain như:

- tỷ lệ phải nằm trong `[0, 100]`;
- số lượng, tuổi, tenure và value không âm;
- tuổi phải nằm trong khoảng `[15, 100]` theo rule hiện tại.

Kết quả thực tế: toàn bộ feature trong `range_specs` có `Status = PASS`, không có `Invalid_Count`.

### 9.4 Out-of-train-range của Engagement

Notebook đếm số giá trị của sáu input engagement nằm dưới min hoặc trên max của phần fit. Nó không clip các giá trị này và ghi chú để preprocessing quyết định.

Đây là kiểm tra hành vi transform, không phải lỗi dữ liệu.

### 9.5 Biểu đồ phân phối

Notebook vẽ histogram cho numeric engineered feature và countplot cho categorical engineered feature. Danh sách gồm đủ 9 feature mới.

Kết quả assertion:

```text
Feature validation passed: plotted all 9 engineered features.
```

### 9.6 Bảng churn rate theo nhóm

Notebook tạo bảng mô tả `Customer_Count` và `Churn_Rate` cho:

- `Purchase_Recency_Group`;
- `Purchase_Frequency_Group`;
- `Service_Call_Group`;
- `Cart_Abandonment_Group`.

Các bảng này chỉ dùng để hiểu business signal sau khi feature đã tạo. Chúng không được dùng để tính feature, không phải model evaluation và không đủ để kết luận feature tốt.

### 9.7 Redundancy bằng correlation

Notebook tính correlation giữa numeric original và numeric engineered, lọc các cặp có absolute correlation từ `0.80` trở lên, rồi hiển thị tối đa 20 cặp.

Mục đích:

```text
phát hiện feature có thể trùng thông tin
-> ghi nhận
-> so sánh trong ML
```

Notebook không tự động xóa feature vì correlation cao.

## 10. Quyết định feature cuối cùng

Có ba trạng thái được dùng:

- `KEEP`: feature gốc được đưa vào baseline;
- `TEST`: feature được phép đưa vào các experiment ML có kiểm soát;
- `EXCLUDE` hoặc `EXCLUDE_PENDING`: không đưa vào export baseline.

### 10.1 KEEP - 20 feature numeric gốc

```text
Age
Membership_Years
Login_Frequency
Session_Duration_Avg
Pages_Per_Session
Cart_Abandonment_Rate
Wishlist_Items
Total_Purchases
Average_Order_Value
Days_Since_Last_Purchase
Discount_Usage_Rate
Returns_Rate
Email_Open_Rate
Customer_Service_Calls
Product_Reviews_Written
Social_Media_Engagement_Score
Mobile_App_Usage
Lifetime_Value
Credit_Balance
```

`Payment_Method_Diversity` là numeric gốc nhưng được đánh dấu `TEST`, không nằm trong KEEP.

### 10.2 TEST - 3 categorical gốc và 9 feature engineered

Categorical gốc:

```text
Gender
Country
Signup_Quarter
```

Numeric/categorical/binary engineered:

```text
Engagement_Score
Purchase_Frequency
Purchase_Recency_Group
Purchase_Frequency_Group
Service_Call_Group
Cart_Abandonment_Group
Estimated_Purchase_Value
Cart_Recency_Interaction
Low_Engagement_Recency
```

Cộng thêm `Payment_Method_Diversity` là 1 numeric gốc ở trạng thái TEST.

Tổng số feature TEST: 13.

### 10.3 EXCLUDE

Trong run hiện tại, chỉ có:

```text
City
```

Các tên target-derived như `Churn_Risk_Score`, `Churn_Score`, `Churn_Probability`, `Churn_Level`, `Churn_Status`, `Churn_Risk` được khai báo là `forbidden_features` để assertion ngăn chúng xuất hiện trong output. Chúng không có trong dataset hiện tại.

## 11. Before và after

Kết quả thực tế:

| Trạng thái | Số lượng |
|---|---:|
| Predictor ban đầu | 24 |
| Numeric predictor ban đầu | 20 |
| Categorical predictor được khai báo | 3 |
| Original được phép export | 23 |
| Engineered được phép export | 9 |
| Tổng feature được phép export | 32 |
| Target `Churned` trong file export | 1 |
| Tổng số cột file export | 33 |

Bảng `Before/After` trong notebook hiển thị các metric ở hai index khác nhau nên có một số ô `NaN`; đó chỉ là cách pandas ghép hai Series khác index, không làm sai các count đã tính. Các count đúng là những con số trong bảng trên.

## 12. Export và kết quả chạy cuối cùng

Notebook tạo thứ tự cột bằng:

```python
export_columns = allowed_features + [TARGET]
```

Sau đó ghi CSV không có index:

```python
df_export.to_csv(OUTPUT_PATH, index=False)
```

Các assertion cuối xác nhận:

- target có trong output;
- feature bị exclude không có trong output;
- feature target-derived bị cấm không có trong output;
- output file tồn tại.

Kết quả chạy cuối cùng:

```text
Exported: D:\Desktop\Projects\customer-churn-analytics\data\processed\customer_churn_feature_engineered.csv
Export shape: (50000, 33)
Leakage status: no target-derived feature is included.
City status: excluded from the baseline feature matrix; incremental predictive value can be evaluated in a later ML experiment.
Transformation status: parameters were fit on an internal demonstration subset only.
Production rule: the final ML pipeline must refit all learned parameters on the official X_train only.
Temporal rule: every time-dependent input must be a pre-prediction snapshot.
Next step: fit the final preprocessing pipeline on X_train once, then transform validation/test data.
```

## 13. Lỗi đã phát hiện và đã sửa

Lần chạy đầu tiên dừng ở cell export vì code dùng:

```python
assert not set(forbidden_features) & set(df_export.columns)
```

nhưng `forbidden_features` chưa được khai báo. Notebook đã được sửa bằng cách khai báo danh sách tên feature target-derived bị cấm trước khi export. Sau đó notebook được execute lại và chạy hoàn tất.

Đây là lỗi biến chưa định nghĩa, không phải lỗi dữ liệu hay leakage trong output.

## 14. Điều notebook đã chứng minh và chưa chứng minh

### Đã chứng minh trong phạm vi hiện tại

- input có schema cần thiết;
- target là nhị phân 0/1;
- không có target-derived predictor theo leakage gate hiện tại;
- 9 engineered feature đã được tạo;
- không có missing, Inf, duplicate column, duplicate row hoặc constant numeric feature sau engineering;
- domain range hiện tại đều PASS;
- City không bị map với nhiều country trong dữ liệu hiện tại;
- output 32 feature + 1 target đã được xuất thành công.

### Chưa được chứng minh

- các feature có làm model tốt hơn hay không;
- threshold hoặc ratio có cải thiện ROC-AUC/Recall/F1 hay không;
- `Total_Purchases` có chắc chắn là order count hay không;
- `Average_Order_Value` có chắc chắn là monetary value/order hay không;
- các biến temporal có chắc chắn được chụp trước prediction time hay không;
- `Engagement_Score` có nên clip về `[0, 1]` hay giữ out-of-range;
- City có incremental predictive value hay không;
- feature nào quan trọng nhất trong mô hình cuối.

## 15. Việc cần làm ở bước tiếp theo

Notebook Preprocessing cần:

1. Chia dữ liệu chính thức thành `X_train`, validation và test theo chiến lược ML đã chọn.
2. Fit mọi tham số học được chỉ trên `X_train`.
3. Encode categorical, gồm cả các group feature mới.
4. Impute nếu pipeline yêu cầu.
5. Scale các numeric feature nếu model yêu cầu.
6. Transform validation/test bằng đúng fitted parameters.
7. Lưu pipeline để inference dùng lại chính xác.

Sau đó Machine Learning cần so sánh tối thiểu:

- baseline chỉ dùng KEEP;
- baseline + từng nhóm TEST;
- các biến numeric so với categorical group tương ứng;
- có và không có các interaction;
- có và không có City trong một experiment riêng.

Chỉ sau các so sánh đó mới được quyết định feature nào từ `TEST` trở thành feature chính thức.

## 16. Kết luận dễ nhớ

Notebook này biến 24 predictor ban đầu thành một feature inventory có kiểm soát:

```text
24 predictor ban đầu
  -> tạo 9 feature mới
  -> loại City khỏi baseline
  -> giữ 23 feature gốc + 9 feature engineered
  -> export 32 feature + Churned
```

Kết quả là một dataset đã được feature engineering, **chưa phải kết quả dự đoán churn**. `KEEP` chỉ là baseline, `TEST` chỉ là ứng viên, và mọi kết luận cuối cùng về hiệu quả phải đến từ bước Machine Learning với train/test pipeline leakage-safe.
