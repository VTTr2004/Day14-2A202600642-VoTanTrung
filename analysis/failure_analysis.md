# Báo cáo phân tích lỗi

## 1. Tổng quan benchmark
- Tổng số case: 50
- Pass/Fail: 43/7
- Model chạy local: kamekichi128/qwen3-4b-instruct-2507:latest
- Điểm judge trung bình: 4.51 / 5.0
- Hit Rate: 96.00%
- MRR: 0.95
- Agreement Rate: 94.00%
- Tổng token Ollama ghi nhận: 119354
- Chi phí API: $0.000000 vì chạy local bằng Ollama
- Quyết định release gate: Rollback

## 2. Phân cụm lỗi
| Cụm lỗi | Số lượng | Vùng nguyên nhân dự kiến |
|---|---:|---|
| incomplete_or_hallucinated | 3 | Prompt sinh câu trả lời / grounding |
| generation_quality | 2 | Tổng hợp câu trả lời |
| retrieval_miss | 2 | Retrieval / viết lại truy vấn |

## 3. Phân tích 5 Whys cho các case tệ nhất
### Case 1: CASE_049 - retrieval_miss
- Câu hỏi: Combo hoặc sản phẩm '[HCM-Nha Trang] WE GO Combo 3N2Đ VMB Bamboo Airways + phòng + bữa sáng và nhiều ưu đãi khác tại Nha Trang' có giá hiện tại là bao nhiêu?
- Điểm: 1.0 / 5.0
- Expected IDs: ['DOC_31F995C8']
- Retrieved IDs: ['DOC_34CD5BCB', 'DOC_9E0D6629', 'DOC_2646CD65']
1. Vì sao case fail? Câu trả lời chưa thỏa hoàn toàn đáp án chuẩn hoặc retrieval target.
2. Vì sao câu trả lời yếu? Context top-1 hoặc bước tổng hợp còn thiếu chi tiết.
3. Vì sao context/tổng hợp thiếu chi tiết? Câu hỏi hard case có nhiều từ khóa giống nhau giữa các combo Vinpearl.
4. Vì sao bị nhiễu giữa các combo? Retriever hiện dùng BM25/lexical nội bộ, chưa có embedding semantic và reranking.
5. Root cause: cần thêm embedding retriever, reranker và prompt bắt model trích dẫn đúng DOC_ID.

### Case 2: CASE_050 - retrieval_miss
- Câu hỏi: Hãy cho biết giá gốc của ưu đãi '[HCM-Nha Trang] WE GO Combo 3N2Đ VMB Bamboo Airways + phòng + bữa sáng và nhiều ưu đãi khác tại Nha Trang'.
- Điểm: 1.0 / 5.0
- Expected IDs: ['DOC_31F995C8']
- Retrieved IDs: ['DOC_34CD5BCB', 'DOC_9E0D6629', 'DOC_0CEA88E5']
1. Vì sao case fail? Câu trả lời chưa thỏa hoàn toàn đáp án chuẩn hoặc retrieval target.
2. Vì sao câu trả lời yếu? Context top-1 hoặc bước tổng hợp còn thiếu chi tiết.
3. Vì sao context/tổng hợp thiếu chi tiết? Câu hỏi hard case có nhiều từ khóa giống nhau giữa các combo Vinpearl.
4. Vì sao bị nhiễu giữa các combo? Retriever hiện dùng BM25/lexical nội bộ, chưa có embedding semantic và reranking.
5. Root cause: cần thêm embedding retriever, reranker và prompt bắt model trích dẫn đúng DOC_ID.

### Case 3: CASE_014 - generation_quality
- Câu hỏi: Hãy cho biết giá gốc của ưu đãi '[Grand World Phú Quốc] Vé Tinh Hoa Việt Nam'.
- Điểm: 1.25 / 5.0
- Expected IDs: ['DOC_EF370EFB']
- Retrieved IDs: ['DOC_EF370EFB', 'DOC_0E1EB692', 'DOC_A3F08343']
1. Vì sao case fail? Câu trả lời chưa thỏa hoàn toàn đáp án chuẩn hoặc retrieval target.
2. Vì sao câu trả lời yếu? Context top-1 hoặc bước tổng hợp còn thiếu chi tiết.
3. Vì sao context/tổng hợp thiếu chi tiết? Câu hỏi hard case có nhiều từ khóa giống nhau giữa các combo Vinpearl.
4. Vì sao bị nhiễu giữa các combo? Retriever hiện dùng BM25/lexical nội bộ, chưa có embedding semantic và reranking.
5. Root cause: cần thêm embedding retriever, reranker và prompt bắt model trích dẫn đúng DOC_ID.

## 4. Kế hoạch cải tiến
- Thêm embedding retriever và reranker để giảm nhầm lẫn giữa các combo có tên gần giống nhau.
- Chuẩn hóa golden dataset bằng review thủ công một phần các case do script sinh.
- Giảm khoảng 30% chi phí/thời gian eval bằng cách chỉ gọi judge thứ hai khi judge thứ nhất nằm vùng không chắc chắn.
- Đưa regression gate vào CI để tự động block khi giảm điểm, hit rate, latency hoặc tăng chi phí.
