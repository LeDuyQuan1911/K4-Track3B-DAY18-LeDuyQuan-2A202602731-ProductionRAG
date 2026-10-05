# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Duy Quân  
**Khóa:** K4 - Track 3B  
**Mã số học viên:** 2A202602731  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6200 | 0.8850 | +0.2650 |
| Answer Relevancy | 0.7100 | 0.8400 | +0.1300 |
| Context Precision | 0.5400 | 0.8100 | +0.2700 |
| Context Recall | 0.6000 | 0.7900 | +0.1900 |

*(Ghi chú: Điểm số cải thiện vượt bậc nhờ kết hợp Hybrid Search BM25 + Dense, Cross-Encoder Reranker lọc top-3 đắt giá nhất và kỹ thuật Contextual Prepend trong M5 giúp giữ ngữ cảnh tài liệu nguồn).*

---

## Bottom-5 Failures

### #1. Xung đột phiên bản quy chế nghỉ phép (Version Conflict)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi 5 năm làm việc liên tục (trích dẫn từ `nghi_phep_nam_v2023.md`).
- **Worst metric:** `context_precision` / `faithfulness`
- **Error Tree:** Output sai thông tin hiện hành → Context chứa tài liệu cũ v2023 thay vì v2024 → Query không chỉ định rõ năm/phiên bản → Hybrid search lấy tài liệu có độ trùng khớp từ khóa cao nhất mà không phân biệt trạng thái hiệu lực.
- **Root cause:** Xung đột phiên bản (Temporal / Versioning Ambiguity). Kho dữ liệu có cả 2 bản `nghi_phep_nam_v2023.md` (hết hiệu lực) và `nghi_phep_nam_v2024.md` (hiện hành). Từ khóa "thâm niên", "ngày phép" xuất hiện ở cả hai, mô hình Bi-Encoder và BM25 cho điểm cao tương đương nhau dẫn đến việc đoạn văn cũ bị trích xuất lên đầu.
- **Suggested fix:** Thêm tầng metadata filter lọc theo `status: active` hoặc `year: 2024`, hoặc trong bước làm giàu M5 (Enrichment), gắn cờ `[LƯU Ý: VĂN BẢN ĐÃ HẾT HIỆU LỰC, THAY THẾ BỞI V2024]` vào đầu các tài liệu cũ để Cross-Encoder và LLM nhận biết.

---

### #2. Truy vấn nhiều bước kết hợp hai nguồn độc lập (Multi-hop Query)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Chỉ tìm thấy thông tin về ngày phép năm (15 ngày cơ bản + thâm niên) nhưng thiếu thông tin dải lương Senior (P3-P4) từ file thang bảng lương.
- **Worst metric:** `context_recall`
- **Error Tree:** Output thiếu 1 vế → Context chỉ chứa đoạn văn quy chế nghỉ phép, thiếu bảng lương → Query gộp 2 ý độc lập vào 1 câu → Tầng retrieval top_k=3 bị chiếm chỗ bởi các đoạn nghỉ phép.
- **Root cause:** Câu hỏi yêu cầu thông tin từ 2 văn bản quy chế hoàn toàn khác nhau (`nghi_phep_nam_v2024.md` và `bang_luong_2024.md`). Khi search câu query dài, các đoạn về "nghỉ phép thâm niên" áp đảo điểm số BM25 và Dense, đẩy các đoạn về bảng lương ra khỏi top 20 candidate.
- **Suggested fix:** Triển khai kỹ thuật Query Decomposition (tách query thành: "Quy định ngày phép theo thâm niên 9 năm" và "Dải lương bậc Senior"), tìm kiếm song song cho từng câu hỏi con rồi merge kết quả trước khi đưa vào Reranker.

---

### #3. Trích xuất thiếu điều kiện ràng buộc trong quy trình mua sắm
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Nêu được cấp Giám đốc phòng ban phê duyệt cho khoản 30 triệu, nhưng bỏ sót điều kiện bắt buộc phải có ít nhất 3 báo giá cạnh tranh.
- **Worst metric:** `context_recall`
- **Error Tree:** Output thiếu điều kiện báo giá → Context trích xuất bảng hạn mức phân quyền nhưng cắt mất phần quy định về thủ tục báo giá → Chunking cắt giữa bảng phân quyền và điều khoản thủ tục.
- **Root cause:** Kích thước chunk nhỏ hoặc cắt theo đoạn khiến bảng phân quyền tài chính nằm ở chunk trước, còn quy định "trên 10 triệu phải có 3 báo giá" lại nằm ở chunk sau.
- **Suggested fix:** Sử dụng Hierarchical Chunking (Parent-Child) trả về toàn bộ Parent Chunk (~2048 ký tự) chứa trọn vẹn cả bảng hạn mức và các điều khoản phụ trợ đi kèm.

---

### #4. Suy luận số học và tính phạt chậm thanh toán (Numeric Reasoning)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Nêu đúng mức phạt 2%/tháng trên số tiền tạm ứng, nhưng tính ra nguyên tháng 300.000 VNĐ mà không quy đổi pro-rata cho 5 ngày quá hạn.
- **Worst metric:** `faithfulness` / `answer_relevancy`
- **Error Tree:** Output suy luận số học chưa chuẩn xác → Context trích dẫn đúng nguyên văn điều khoản phạt 2%/tháng và thời hạn 15 ngày → LLM không tự suy diễn phép tính chia theo số ngày thực tế quá hạn.
- **Root cause:** Các mô hình ngôn ngữ vừa và nhỏ (như `gpt-4o-mini`) dễ mắc lỗi suy luận số học nhiều bước (tính số ngày quá hạn = 20 - 15 = 5 ngày, rồi nhân tỉ lệ 5/30 với lãi suất tháng).
- **Suggested fix:** Tối ưu hóa System Prompt bằng kỹ thuật Chain-of-Thought (CoT): "Với các câu hỏi có thời gian và số tiền phạt, hãy trình bày rõ các bước: (1) Số ngày quá hạn, (2) Công thức áp dụng, (3) Kết quả tính toán chi tiết."

---

### #5. Quy chế xung đột giữa hai chức danh trong chương trình Onboarding
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?
- **Expected:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Got:** Trả lời đúng việc Quản lý trực tiếp không được làm mentor, nhưng chưa khẳng định dứt khoát việc mentor và buddy phải là 2 cá nhân độc lập.
- **Worst metric:** `answer_relevancy`
- **Error Tree:** Output trả lời thiếu 1 trong 2 vế câu hỏi kép → Context có đoạn văn quy chế nhưng câu chữ mô tả nằm ở 2 mục con khác nhau trong tài liệu `mentor_buddy.md`.
- **Root cause:** Câu hỏi kép có 2 dấu hỏi chấm ("không?... không?"). Mô hình LLM khi sinh câu trả lời có xu hướng tập trung trả lời câu hỏi ở vế sau mà lướt qua điều kiện ở vế trước.
- **Suggested fix:** Cải tiến prompt generation yêu cầu: "Nếu câu hỏi có nhiều vế, bắt buộc phải tách thành từng đầu mục đánh số tương ứng (1), (2) để trả lời trọn vẹn từng ý."

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Sai. Mô hình trả lời 5 năm (theo quy chế cũ 2023), trong khi quy chế hiện hành 2024 quy định chỉ cần 3 năm.
2. **Context đúng?** → Sai. Đoạn văn được xếp hạng đầu bảng (Rank 1) là đoạn trích từ file `nghi_phep_nam_v2023.md`.
3. **Query rewrite OK?** → Câu hỏi người dùng ở dạng khái quát chung ("Thâm niên bao nhiêu năm..."), không chứa từ khóa "2024" hay "mới nhất".
4. **Fix ở bước:**  
   - **Tầng M1/M5:** Bổ sung metadata năm ban hành (`effective_date: 2024`, `status: active`) và dùng Contextual Prepend để ghi rõ trạng thái tài liệu vào đầu chunk.
   - **Tầng M2/M3:** Áp dụng bộ lọc Metadata Filter ưu tiên các tài liệu đang có hiệu lực (`status == active`), hoặc reranker prompt chú ý đến tính cập nhật của thông tin.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Triển khai Metadata Filtering theo trạng thái tài liệu:** Loại bỏ hoặc đánh dấu giảm điểm các văn bản thuộc nhóm `superseded` (v2023, v1.0).
- **Tích hợp Query Decomposition & Multi-Query Expansion:** Giải quyết triệt để các câu hỏi đa bước (Multi-hop) như câu hỏi kết hợp chính sách phép và thang bảng lương.
- **Dynamic Context Routing:** Tự động điều chỉnh `top_k` của reranker linh hoạt tùy theo độ phức tạp của câu hỏi.
