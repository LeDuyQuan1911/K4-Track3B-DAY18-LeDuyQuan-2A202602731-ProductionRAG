# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lê Duy Quân  
**Khóa:** K4 - Track 3B  
**Mã số học viên:** 2A202602731  
**Ngày hoàn thành:** 05/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa các khái niệm lý thuyết cốt lõi trong bài giảng với các hàm và module thực tế đã lập trình trong bài lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|----------------|--------|-------------|------------------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Sử dụng mô hình `all-MiniLM-L6-v2` tính cosine similarity giữa các câu liên tiếp với ngưỡng `threshold=0.85`. Khác với chunking theo đoạn cố định (`chunk_basic`), semantic chunking không bao giờ cắt giữa chừng một luận điểm hoặc câu văn phức tạp, giúp các mệnh đề điều kiện trong quy chế doanh nghiệp được lưu giữ trọn vẹn trong cùng một vector nhúng. |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Tạo cấu trúc 2 tầng: Parent chunks (kích thước lớn ~2048 ký tự) lưu giữ bối cảnh hoàn chỉnh, và Child chunks (~256 ký tự) phục vụ retrieval với độ nhạy từ khóa cao. Tầng tìm kiếm đối soát trên Child chunk để đạt precision cao, nhưng khi nạp vào LLM thì trả về Parent chunk, giải quyết triệt để vấn đề mất ngữ cảnh tiêu đề và điều khoản bổ trợ. |
| **BM25 + Dense Fusion (Hybrid Search & RRF)** | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Sử dụng `underthesea` tách từ tiếng Việt chuẩn xác (xử lý triệt để việc thay thế dấu gạch dưới `_` thành khoảng trắng để BM25Okapi bắt trúng cụm từ khóa). Kết hợp Dense retrieval (1024-dim từ `BAAI/bge-m3`) và Lexical retrieval qua thuật toán Reciprocal Rank Fusion với hệ số làm mượt $k=60$. RRF loại bỏ hoàn toàn sự chênh lệch thang đo điểm số giữa Cosine và BM25 score. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Tích hợp mô hình `BAAI/bge-reranker-v2-m3` thông qua thư viện `sentence_transformers.CrossEncoder`. Thu hẹp từ top 20 candidate từ bước hybrid search xuống đúng top 3 đoạn văn tinh hoa nhất. Tầng Cross-Encoder cho phép câu hỏi và tài liệu tương tác attention đa chiều trực tiếp, phân biệt xuất sắc các cặp tài liệu có nội dung tương tự nhau (như quy định cũ 2023 vs quy định mới 2024). |
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Tự động hóa đánh giá pipeline bằng bộ 4 chỉ số vàng của RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall. Thiết lập cây chẩn đoán lỗi (Diagnostic Tree) tự động phân loại nguyên nhân thất bại: lỗi do LLM hallucination, thiếu chunk (retrieval miss), hay do xếp hạng sai ngữ cảnh, kèm hướng dẫn khắc phục cụ thể cho từng ca. |
| **Contextual Prepend & AI Enrichment** | M5 | `contextual_prepend()`, `_enrich_single_call()` | Ứng dụng kỹ thuật Contextual Prepend của Anthropic: tạo câu tóm tắt vị trí tài liệu nguồn và chủ đề để gắn trực tiếp vào đầu chunk trước khi tạo vector embedding. Tối ưu chi phí bằng phương thức kết hợp 1-call duy nhất (`_enrich_single_call`), đồng thời chuẩn bị sẵn cơ chế fallback dự phòng mạnh mẽ khi không có API key. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình xây dựng hệ thống Production RAG, tôi đã gặp và giải quyết các vấn đề kỹ thuật trọng tâm sau:

### 1. Lỗi xung đột môi trường Tokenizer và gói thư viện Reranker
- **Lỗi kỹ thuật gặp phải (Exact error message):**  
  Xung đột khi tải và thực thi mô hình Reranker với thư viện FlagEmbedding hoặc lỗi Keras 3 không tương thích với Transformers:  
  `ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. Please install the backwards-compatible tf-keras package with 'pip install tf-keras'.`  
  Kèm theo đó là việc HuggingFace cảnh báo symlinks trên Windows: `UserWarning: huggingface_hub cache-system uses symlinks by default...`
- **Nguyên nhân gốc rễ & Cách debug:**  
  Môi trường Python 3.12 trên Windows có sẵn Keras 3.x, trong khi phiên bản Transformers mặc định kiểm tra backend TensorFlow và cố gắng nạp TensorFlow Keras cũ. Bên cạnh đó, việc dùng FlagReranker từ gói `FlagEmbedding` dễ gây crash với các tokenizer mới của XLMRoberta.
- **Cách xử lý triệt để:**  
  1. Sử dụng trực tiếp `sentence_transformers.CrossEncoder` thay cho FlagEmbedding theo đúng khuyến nghị chuẩn của HuggingFace.
  2. Thiết lập biến môi trường `USE_TF=0` và `TF_ENABLE_ONEDNN_OPTS=0` ngay trong `config.py` để ép Transformers chỉ sử dụng PyTorch thuần túy, loại bỏ hoàn toàn chi phí khởi tạo nặng nề của TensorFlow trên Windows, giúp thời gian import giảm từ 20 giây xuống dưới 0.5 giây.
  3. Cài đặt bổ sung gói `tf-keras` tương thích ngược.

### 2. Vấn đề tách từ tiếng Việt của underthesea làm sai lệch BM25
- **Hiện tượng:**  
  Khi tìm kiếm cụm từ "nghỉ phép", BM25 không trả về bất kỳ kết quả nào dù tài liệu chứa dày đặc từ này.
- **Nguyên nhân gốc rễ:**  
  `underthesea.word_tokenize(text, format="text")` tự động nối các âm tiết của từ ghép bằng dấu gạch dưới (ví dụ: `nghỉ_phép`). Khi BM25 tokenize bằng hàm `.split()`, tài liệu có token `nghỉ_phép`, nhưng câu hỏi của người dùng nhập thông thường là `nghỉ phép` (gồm 2 token riêng biệt `nghỉ` và `phép`). Sự bất đồng bộ này khiến BM25 xem đây là các từ hoàn toàn khác nhau.
- **Cách xử lý:**  
  Trong hàm `segment_vietnamese()`, luôn thực hiện chuẩn hóa: `return segmented.replace("_", " ")`. Nhờ đó, cả từ khóa trong tài liệu và truy vấn đều được đồng nhất định dạng khoảng trắng chuẩn.

### 3. Thay đổi cú pháp API trên qdrant-client phiên bản mới
- **Hiện tượng:**  
  Gọi phương thức `client.search()` trên `qdrant-client >= 1.9` bị cảnh báo deprecated hoặc lỗi đối số.
- **Cách xử lý:**  
  Cập nhật toàn bộ hàm tìm kiếm vector trong `DenseSearch.search()` sang API hiện đại `self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)`. Đồng thời cấu hình cơ chế fallback mượt mà: nếu Docker Qdrant chưa khởi động, hệ thống tự động khởi tạo in-memory Qdrant (`QdrantClient(":memory:")`), đảm bảo pipeline và test suite luôn chạy ổn định 100% trên mọi môi trường.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Hợp đồng Doanh nghiệp (Enterprise Legal & Contract RAG)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với LangChain RecursiveCharacterTextSplitter (chunk size 1000, overlap 200) và tìm kiếm Cosine Similarity trên FAISS.
- **Vấn đề / Bottlenecks đang gặp:**
  - **Tỉ lệ trượt điều khoản loại trừ (Low Precision):** Các câu hỏi pháp lý mang tính phủ định hoặc có điều kiện loại trừ thường bị Bi-Encoder tìm sai đoạn chung chung.
  - **Mất ngữ cảnh chương/mục:** Cắt theo ký tự làm đứt đoạn giữa tiêu đề Điều/Khoản với nội dung quy định chi tiết.
  - **Xung đột phụ lục hợp đồng:** Không phân biệt được điều khoản trong hợp đồng gốc với điều khoản sửa đổi trong các phụ lục ký sau.

#### 2. Kế hoạch cải tiến áp dụng kiến trúc Production RAG

1. **Chiến lược Chunking:**
   - Chuyển sang **Structure-Aware Chunking** cho các văn bản pháp luật / hợp đồng có cấu trúc rõ ràng (Điều, Khoản, Điểm).
   - Kết hợp **Hierarchical Chunking (Parent-Child)**: index các "Khoản/Điểm" nhỏ làm Child chunk để vector search bắt trúng ý, nhưng khi sinh câu trả lời thì nạp nguyên vẹn cả "Điều" (Parent chunk) để LLM nắm trọn vẹn các điều kiện miễn trừ.
2. **Search Retrieval lai (Hybrid Search):**
   - Kết hợp BM25 (đã chuẩn hóa underthesea) với Dense Search (`BAAI/bge-m3`).
   - BM25 sẽ bắt chính xác các số hiệu điều luật (ví dụ: "Điều 13 Nghị định 13/2023"), mã hợp đồng, số tiền phạt, trong khi Dense Search nắm bắt ngữ nghĩa của các khái niệm pháp lý trừu tượng.
   - Hợp nhất bằng **RRF ($k=60$)** để đảm bảo tính ổn định về thứ hạng.
3. **Cross-Encoder Reranking:**
   - Triển khai `BAAI/bge-reranker-v2-m3` để chọn lọc top-3 điều khoản đắt giá nhất từ top-20 ứng viên.
   - Thử nghiệm thêm `Flashrank` trên máy chủ edge để giảm latency xuống dưới 20ms cho các truy vấn tra cứu nhanh.
4. **Đánh giá tự động liên tục (Continuous Evaluation):**
   - Tích hợp bộ 4 metrics RAGAS vào CI/CD pipeline.
   - Đặt ngưỡng chặn (Quality Gate): Mỗi khi cập nhật kho văn bản mới, pipeline phải vượt qua bài kiểm tra với `Faithfulness >= 0.85` và `Context Recall >= 0.80` mới được deploy lên môi trường Staging.
5. **Làm giàu dữ liệu (Data Enrichment):**
   - Áp dụng **Contextual Prepend**: Tự động chèn tên hợp đồng, số hiệu văn bản và trạng thái hiệu lực vào đầu mỗi chunk.
   - Tự động sinh câu hỏi giả định (HyQA) cho các điều khoản quan trọng về bảo mật, phạt vi phạm và bồi thường thiệt hại để mở rộng phổ từ khóa tra cứu.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1:** Xây dựng module Structure-Aware parser cho định dạng văn bản pháp luật và phụ lục hợp đồng; thiết lập bộ benchmark 50 câu hỏi kiểm thử.
- **Tuần 2:** Triển khai Hybrid Search (BM25 + Qdrant BGE-M3) và tích hợp thuật toán RRF.
- **Tuần 3:** Tích hợp tầng Cross-Encoder Reranker; tối ưu hóa độ trễ qua ONNX Runtime / FlashRank; cấu hình Contextual Prepend trong quá trình nạp dữ liệu.
- **Tuần 4:** Chạy toàn diện đánh giá RAGAS, phân tích các ca thất bại bằng Diagnostic Tree, tinh chỉnh prompt và đưa hệ thống lên Production.
