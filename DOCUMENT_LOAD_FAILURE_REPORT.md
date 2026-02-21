# Document Load Failure - Analysis Report

**Date:** October 8, 2025
**System:** Contextprime - Docker Deployment
**Severity:** Medium (API endpoint crash, but processing succeeded)

---

## Executive Summary

A document was successfully processed (662 chunks from a 61,676-word PDF) but the Web UI failed to display the document list due to a type mismatch in the API response layer. The document data is intact in memory; only the listing endpoint is affected.

---

## Issue Details

### Symptom
- **HTTP Status:** 500 Internal Server Error
- **Endpoint:** `GET /api/documents`
- **Error Type:** `pydantic_core.ValidationError`
- **User Impact:** Web UI cannot display processed documents list

### Error Message
```
ValidationError: 1 validation error for DocumentListResponse
documents.0
  Input should be a valid dictionary or instance of DocumentInfo
  [type=model_type, input_value=StoredDocument(info=Docum...Processed successfully'),
  input_type=StoredDocument]
```

### Root Cause
**Type mismatch in processing service return value**

**Location:** `/Users/simonkelly/SUPER_RAG/doctags_rag/src/api/services/processing_service.py:93-98`

```python
def list_documents(self) -> List[DocumentInfo]:
    """Return processed documents sorted by upload time (newest first)."""
    with self._lock:
        docs = list(self._documents.values())

    return sorted(docs, key=lambda item: item.info.uploaded_at, reverse=True)
```

**Problem:** The method returns `List[StoredDocument]` but declares `List[DocumentInfo]` as return type.

**Data Flow:**
1. `self._documents` stores `Dict[str, StoredDocument]` (line 43)
2. `.values()` returns `StoredDocument` objects (line 96)
3. API router receives `List[StoredDocument]` (documents.py:65)
4. Pydantic expects `List[DocumentInfo]` for validation (responses.py:124)
5. Validation fails → HTTP 500 error

---

## Document Processing Status

### ✅ Successfully Processed
- **Filename:** tmp3x7mfuhz.pdf
- **Document ID:** ad8e8c8a51896863
- **Elements Parsed:** 12,012
- **Word Count:** 61,676
- **Character Count:** 417,873
- **Chunks Created:** 662
- **Processing Time:** 6.88 seconds
- **Tags Generated:** 12,013 DocTags

### Document Storage
Document is correctly stored in memory at `StoredDocument` dataclass with:
- ✅ DocumentInfo metadata
- ✅ 662 chunk objects
- ✅ Markdown conversion
- ✅ Text preview (2000 chars)
- ✅ DocTags structure
- ✅ Success message

---

## Technical Analysis

### Architecture Context

The API uses a three-layer response model:

```
┌─────────────────────────────────────────┐
│  StoredDocument (Internal Storage)      │
│  ├─ info: DocumentInfo                  │
│  ├─ chunks: List[Dict]                  │
│  ├─ markdown: Optional[str]             │
│  └─ text_preview: Optional[str]         │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  DocumentInfo (Public Metadata)         │
│  ├─ id, filename, file_type             │
│  ├─ num_chunks, processing_time         │
│  └─ status, uploaded_at, metadata       │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  DocumentListResponse (API Response)    │
│  ├─ success: bool                       │
│  ├─ documents: List[DocumentInfo] ⚠️    │
│  └─ total, page, page_size              │
└─────────────────────────────────────────┘
```

**Design Intent:**
- `StoredDocument` = Rich internal representation (full chunks, markdown)
- `DocumentInfo` = Lightweight metadata for list views
- `DocumentListResponse` = Paginated API response

**Actual Behavior:**
The `list_documents()` method returns the full `StoredDocument` objects instead of extracting the lightweight `DocumentInfo` metadata, causing type validation failure.

---

## Fix Required

### Solution
Extract the `.info` attribute from each `StoredDocument`:

**File:** `doctags_rag/src/api/services/processing_service.py`
**Line:** 93-98

**Current Code:**
```python
def list_documents(self) -> List[DocumentInfo]:
    """Return processed documents sorted by upload time (newest first)."""
    with self._lock:
        docs = list(self._documents.values())

    return sorted(docs, key=lambda item: item.info.uploaded_at, reverse=True)
```

**Corrected Code:**
```python
def list_documents(self) -> List[DocumentInfo]:
    """Return processed documents sorted by upload time (newest first)."""
    with self._lock:
        docs = [doc.info for doc in self._documents.values()]

    return sorted(docs, key=lambda item: item.uploaded_at, reverse=True)
```

**Changes:**
1. Line 96: Extract `.info` attribute → `[doc.info for doc in ...]`
2. Line 98: Remove `.info` from sort key → `item.uploaded_at` (not `item.info.uploaded_at`)

---

## Testing Verification

### Expected Behavior After Fix
1. ✅ `GET /api/documents` returns HTTP 200
2. ✅ Response contains `DocumentListResponse` with valid JSON
3. ✅ Web UI displays document list correctly
4. ✅ Document metadata shows: filename, chunk count, processing time

### Test Cases
```bash
# 1. Upload document
curl -X POST http://localhost:8000/api/documents \
  -F "file=@test.pdf" \
  -F "settings={...}"

# 2. List documents (should succeed after fix)
curl http://localhost:8000/api/documents

# 3. Verify response structure
{
  "success": true,
  "documents": [
    {
      "id": "ad8e8c8a51896863",
      "filename": "test.pdf",
      "file_type": "pdf",
      "num_chunks": 662,
      "processing_time": 6.88,
      "status": "completed",
      "uploaded_at": "2025-10-08T14:26:32Z"
    }
  ],
  "total": 1
}
```

---

## Impact Assessment

### Current Impact
- **Severity:** Medium
- **Scope:** Document listing endpoint only
- **Data Loss:** None (documents stored correctly)
- **Workaround:** Access documents via direct ID: `GET /api/documents/{document_id}`

### Services Affected
| Service | Status | Notes |
|---------|--------|-------|
| Document Upload | ✅ Working | Processing pipeline functions correctly |
| Document Processing | ✅ Working | 662 chunks created successfully |
| Document Storage | ✅ Working | Data persisted in memory |
| Document Listing | ❌ **Broken** | Type validation fails |
| Document Detail | ✅ Working | Individual document retrieval works |
| Search API | ✅ Working | Not affected |
| Agentic API | ✅ Working | Not affected |

---

## Related Issues

### Similar Pattern Detected
The `get_document()` method (line 100-103) returns `StoredDocument` correctly and is consumed properly by the detail endpoint because it explicitly handles the nested structure:

```python
# documents.py:80-89 (working correctly)
stored = await run_in_threadpool(state.processing_service.get_document, document_id)
return DocumentDetailResponse(
    document=stored.info,  # ✅ Extracts .info correctly
    chunks=chunks,
    ...
)
```

The list endpoint should follow the same pattern.

---

## Prevention Recommendations

### 1. Type Safety Enforcement
Enable mypy strict mode to catch return type mismatches:
```ini
# pyproject.toml or mypy.ini
[mypy]
strict = true
warn_return_any = true
```

This would have caught: `error: Returning type List[StoredDocument] but expected List[DocumentInfo]`

### 2. Unit Tests
Add API contract tests:
```python
def test_list_documents_returns_document_info():
    service = ProcessingService()
    # ... upload test document
    docs = service.list_documents()
    assert isinstance(docs, list)
    assert all(isinstance(d, DocumentInfo) for d in docs)
```

### 3. Integration Tests
Add end-to-end API tests:
```python
def test_list_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "documents" in data
```

---

## System Context

### Container Status
```
NAME             IMAGE                   STATUS
doctags-app      super_rag-app           Up (port 8000)
doctags-neo4j    neo4j:5.15.0            Up (healthy)
doctags-qdrant   qdrant/qdrant:v1.14.0   Up (healthy)
```

### Recent Activity
- Document uploaded via Web UI (POST /api/documents)
- Processing completed successfully (6.88s)
- List endpoint called (GET /api/documents)
- **Error triggered** (500 Internal Server Error)
- Status endpoint remains healthy (GET /api/status → 200 OK)

---

## Logs Extract

### Successful Processing
```
2025-10-08 14:26:25 | INFO | Processing upload with chunk_size=1000 overlap=200
2025-10-08 14:26:25 | INFO | Parsing pdf document: tmp3x7mfuhz.pdf
2025-10-08 14:26:31 | INFO | Parsed: 12012 elements, 61676 words
2025-10-08 14:26:32 | INFO | Created 662 chunks from document ad8e8c8a51896863
2025-10-08 14:26:32 | INFO | Completed in 6.88s
POST /api/documents → 201 Created ✅
```

### List Endpoint Failure
```
GET /api/documents → 500 Internal Server Error ❌
ValidationError: Input should be a valid dictionary or instance of DocumentInfo
  input_type=StoredDocument
```

---

## Appendices

### A. Data Structures

**StoredDocument** (internal):
```python
@dataclass
class StoredDocument:
    info: DocumentInfo
    chunks: List[Dict[str, Any]]
    markdown: Optional[str]
    text_preview: Optional[str]
    doctags: Optional[Dict[str, Any]]
    message: Optional[str] = None
```

**DocumentInfo** (public API):
```python
class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    num_chunks: int
    processing_time: float
    status: ProcessingStatus
    uploaded_at: datetime
    metadata: Dict[str, Any]
```

---

## Summary

**Problem:** API type mismatch in document listing
**Cause:** Method returns `StoredDocument` instead of `DocumentInfo`
**Fix Complexity:** Simple (2-line change)
**Data Integrity:** Unaffected (document successfully stored)
**Priority:** Medium (workaround available via direct document access)

---

**Report Generated:** October 8, 2025, 15:30 UTC
**System Version:** 1.0.0
**Docker Environment:** Development
