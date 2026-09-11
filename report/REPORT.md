# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 09/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Thay đổi **KHOA** để tạo tên output theo yêu cầu bài thực hành; các logic và cấu hình khác được giữ nguyên.
> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - `class_id`: `468`
  - `class_name`: `cab`
  - `rank`: `1`
  - `score`: `0.510915`
  - `taxonomy_name`: `ImageNet-1K`

- Record này mô tả toàn ảnh như thế nào?
  - Model dự đoán toàn ảnh thuộc lớp `cab` với score cao nhất là `0.510915`. Đây là dự đoán ở cấp độ toàn ảnh, không cung cấp vị trí cụ thể của đối tượng.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Class list được xác định bởi taxonomy/dataset mà checkpoint được huấn luyện trên. Với checkpoint này, taxonomy là `ImageNet-1K`.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` giúp định danh lớp một cách ổn định.
  - `class_name` giúp con người đọc và hiểu lớp.
  - `taxonomy_name` cho biết lớp thuộc hệ phân loại nào.
  - Giữ cả ba giúp đối chiếu chính xác giữa model, dữ liệu và hệ thống sử dụng kết quả.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Guideline cần quy định rõ cách chọn nhãn đại diện cho toàn ảnh khi có nhiều chủ thể, đặc biệt là tiêu chí xác định chủ thể/lớp chính và cách xử lý trường hợp nhiều chủ thể có mức độ quan trọng tương đương.

- Vì sao model score không phải ground truth?
  - `score` chỉ là mức độ tin cậy của model đối với dự đoán. Nó không xác nhận dự đoán là đúng. Ground truth phải được xác định từ nhãn do con người tạo ra theo guideline và được kiểm tra qua quy trình QC.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - `class_name`: `person`
  - `score`: `0.912625`
  - `bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`
  - `bbox_width`: `113.58`
  - `bbox_height`: `279.68`

- Diễn giải vị trí box bằng lời:
  - Box bao quanh người đứng ở khu vực bên phải và gần trung tâm ảnh, kéo từ khoảng `x = 385` đến `499` và `y = 69` đến `349`.

- So sánh số prediction ở hai threshold:
  - Ở threshold `0.35`: có `53` prediction.
  - Nếu tăng threshold lên `0.50`: còn `33` prediction.
  - Như vậy số prediction giảm `20` record khi tăng threshold, do các prediction có score thấp hơn `0.50` bị loại.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Threshold thấp giữ lại nhiều prediction hơn nên độ bao phủ cao hơn, nhưng reviewer phải kiểm tra nhiều prediction hơn và có thể gặp nhiều false positive.
  - Threshold cao giảm số prediction cần review, giúp giảm khối lượng QC nhưng có nguy cơ bỏ sót các object có score thấp.

- Đề xuất một quy tắc box chặt:
  - Box nên bao phủ toàn bộ object nhưng chỉ đến sát biên ngoài của object, hạn chế tối đa phần background và không cắt mất phần nhìn thấy của object.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định rõ có annotate phần object bị che khuất hay chỉ phần nhìn thấy.
  - Với object bị cắt bởi mép ảnh, cần quy định có giữ box sát mép ảnh hay đánh dấu trường hợp đặc biệt.
  - Các trường hợp không thể xác định rõ biên object nên được đưa vào escalation để reviewer quyết định thay vì annotator tự suy đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: `kitchen-001`
  - `class_name`: `person`
  - `score`: `0.899318`
  - `polygon_point_count`: `348`
  - Một phần `polygon_xy`: `[[446, 70], [445, 71], [444, 71], [443, 72], ...]`

- Polygon bổ sung chi tiết gì so với box?
  - Box chỉ xác định vùng hình chữ nhật bao quanh object.
  - Polygon mô tả sát hơn đường biên thực tế của object, giúp phân biệt phần object với background.
  - Với ảnh này, mask của `person` bám theo hình dáng cơ thể thay vì bao trọn toàn bộ hình chữ nhật.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` dùng để phân biệt từng object cụ thể trong cùng một ảnh, ví dụ `kitchen-001`, `kitchen-002`, ...
  - Đây không phải `class_id`: `class_id` xác định loại object, còn `instance_id` xác định từng instance cụ thể của loại đó.

- Đề xuất một quy tắc biên mask:
  - Mask phải bám sát phần nhìn thấy của object, không ăn sang background và không bỏ sót phần object có thể xác định rõ.
  - Với các chi tiết nhỏ, chỉ đưa vào mask khi có thể xác định biên rõ ràng từ ảnh.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định annotate theo phần nhìn thấy hay ước lượng toàn bộ object khi bị che khuất.
  - Cần quy định cách xử lý vùng biên mờ, vật thể chồng lấn hoặc tiếp xúc với nhau.
  - Nếu không thể xác định rõ ranh giới object, annotator nên escalation cho reviewer thay vì tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng
`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
|-|-|-|-|-|
| Phân loại ảnh | 1 ảnh -> 1 nhãn lớp (`class_id`, `class_name`, `taxonomy_name`) | Ảnh có nhiều chủ thể hoặc nội dung không rõ lớp chính; prediction score thấp có thể cho thấy model không chắc chắn | Chọn đúng lớp theo guideline; không suy đoán khi không đủ căn cứ; đánh dấu/escalate trường hợp mơ hồ | Kiểm tra lớp có phù hợp toàn ảnh và đúng taxonomy; xem các trường hợp mơ hồ hoặc dễ nhầm lớp |
| Phát hiện vật thể | Mỗi object -> 1 class + `bbox_xyxy` | Box có thể quá rộng/hẹp, bỏ sót object hoặc chứa nhiều background; object nhỏ, bị che khuất hoặc cắt mép ảnh dễ gây sai lệch | Vẽ box sát phần object nhìn thấy; đảm bảo đúng class; xử lý theo guideline khi bị che khuất/cắt mép | Kiểm tra box có bao phủ đúng object, không thừa background, không bỏ sót object và class có chính xác |
| Instance segmentation | Mỗi instance -> 1 `instance_id` + class + polygon/mask | Biên mask khó xác định ở vùng mờ, tiếp xúc/chồng lấn hoặc bị che khuất; các chi tiết nhỏ dễ bị ăn vào background hoặc bỏ sót | Vẽ polygon bám sát biên phần object nhìn thấy; không tự suy đoán vùng không rõ; escalate khi không xác định được biên | Kiểm tra mask có bám đúng hình dạng object, tách đúng từng instance và xử lý nhất quán các vùng che khuất/chồng lấn |

## 5. An toàn dữ liệu

* Một quy tắc bảo vệ dữ liệu:
  * Chỉ sử dụng dữ liệu đúng phạm vi được giao, không tự ý sao chép, chia sẻ hoặc đưa dữ liệu ra ngoài phạm vi công việc.

* Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  * Người phụ trách bài thực hành / Reviewer.

## 6. Danh sách bằng chứng

* [x] `classification_predictions.json`
* [x] `detection_predictions.json`
* [x] `segmentation_predictions.json`
* [x] `IMAGE_ATTRIBUTION.md`
* [x] `visuals/classification_top5.png`
* [x] `visuals/detection_predictions.png`
* [x] `visuals/segmentation_prediction.png`
* [x] Ô validation cuối notebook báo `PASS`.
* [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

