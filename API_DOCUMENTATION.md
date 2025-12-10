# Kalaam AI Backend API Documentation

## Base URL
```
http://localhost:8000
```

## Response Format
All endpoints (except `/auth/token`) return responses in the following standard format:

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "message": "Success message",
  "status_code": 200
}
```

### Error Response
```json
{
  "success": false,
  "data": null,
  "error": "Error description",
  "message": null,
  "status_code": 400
}
```

---

## Authentication Endpoints

### 1. Register User
**Endpoint:** `POST /auth/register`

**Authentication Required:** No

**Request Body:**
```json
{
  "username": "string (min 3 characters)",
  "password": "string (min 8 characters)"
}
```

**Success Response (201):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "username": "john_doe"
  },
  "error": null,
  "message": "User registered successfully",
  "status_code": 201
}
```

**Error Responses:**

**Username already exists (400):**
```json
{
  "success": false,
  "data": null,
  "error": "Username already exists",
  "message": null,
  "status_code": 400
}
```

**Password too short (400):**
```json
{
  "success": false,
  "data": null,
  "error": "The password must be greater than 8 characters!",
  "message": null,
  "status_code": 400
}
```

---

### 2. Login (for Frontend - Use This)
**Endpoint:** `POST /auth/login`

**Authentication Required:** No

**Request Body:**
```json
{
  "username": "john_doe",
  "password": "password123"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  },
  "error": null,
  "message": "Login successful",
  "status_code": 200
}
```

**Error Response (401):**
```json
{
  "success": false,
  "data": null,
  "error": "Invalid credentials",
  "message": null,
  "status_code": 401
}
```

**How to Use the Token:**
After successful login, store the `access_token` from the response. Include it in the `Authorization` header for all protected routes:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

### 3. OAuth2 Token (for Swagger UI - Don't Use)
**Endpoint:** `POST /auth/token`

**Note:** This endpoint is for Swagger UI OAuth2 authentication only. Frontend should use `/auth/login` instead.

---

## Analysis Endpoints (Protected)

**⚠️ All analysis endpoints require authentication.**

Include the JWT token in the Authorization header:
```
Authorization: Bearer <your_access_token>
```

### 4. Upload and Analyze File
**Endpoint:** `POST /api/analyze`

**Authentication Required:** Yes (Bearer Token)

**Request Type:** `multipart/form-data`

**Request Body:**
- `file`: Audio or video file (mp3, mp4, wav, avi)
- Max size: 20MB

**Example using JavaScript (fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/analyze', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${accessToken}`
  },
  body: formData
});

const data = await response.json();
```

**Example using Axios:**
```javascript
const formData = new FormData();
formData.append('file', file);

const response = await axios.post('http://localhost:8000/api/analyze', formData, {
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'multipart/form-data'
  }
});
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "analysis_id": 1,
    "file_name": "sample.mp4",
    "file_type": "video",
    "transcript": "Hello, this is a sample transcript...",
    "wpm_data": [
      {
        "start_time": 0.0,
        "end_time": 2.0,
        "word_count": 5,
        "wpm": 150
      },
      {
        "start_time": 2.0,
        "end_time": 4.0,
        "word_count": 6,
        "wpm": 180
      }
    ],
    "filler_word_analysis": {
      "fillers": [
        {
          "text": "um",
          "timestamp": 1234,
          "type": "single"
        }
      ],
      "filler_count": 5,
      "total_words": 120,
      "filler_percentage": 4.17
    },
    "loudness_analysis": {
      "intervals": [
        {
          "start_time": 0.0,
          "end_time": 1.0,
          "rms_db": -12.5,
          "lufs": -16.3
        }
      ],
      "statistics": {
        "average_rms_db": -15.2,
        "average_lufs": -18.5,
        "peak_rms_db": -8.3,
        "peak_lufs": -12.1
      }
    },
    "created_at": "2025-12-11T10:30:00"
  },
  "error": null,
  "message": "Analysis completed and saved successfully",
  "status_code": 200
}
```

**Error Responses:**

**Invalid file type (400):**
```json
{
  "success": false,
  "data": null,
  "error": "File type .pdf not supported. Allowed: .mp4, .mp3, .wav, .avi",
  "message": null,
  "status_code": 400
}
```

**File too large (413):**
```json
{
  "success": false,
  "data": null,
  "error": "File size exceeds 20MB limit",
  "message": null,
  "status_code": 413
}
```

**Unauthorized (401):**
```json
{
  "success": false,
  "data": null,
  "error": "Invalid or expired token",
  "message": null,
  "status_code": 401
}
```

**Analysis failed (500):**
```json
{
  "success": false,
  "data": null,
  "error": "Analysis failed: Transcription error message",
  "message": null,
  "status_code": 500
}
```

---

### 5. Get Analysis by ID
**Endpoint:** `GET /api/analyze/{analysis_id}`

**Authentication Required:** Yes (Bearer Token)

**Path Parameters:**
- `analysis_id`: Integer - The ID of the analysis to retrieve

**Example Request:**
```javascript
const response = await fetch('http://localhost:8000/api/analyze/1', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});

const data = await response.json();
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "file_name": "sample.mp4",
    "file_type": "video",
    "status": "completed",
    "transcript": "Hello, this is a sample transcript...",
    "wpm_data": [ ... ],
    "error_message": null,
    "created_at": "2025-12-11T10:30:00",
    "updated_at": "2025-12-11T10:31:00"
  },
  "error": null,
  "message": "Analysis retrieved successfully",
  "status_code": 200
}
```

**Error Responses:**

**Analysis not found (404):**
```json
{
  "success": false,
  "data": null,
  "error": "Analysis not found",
  "message": null,
  "status_code": 404
}
```

**Note:** Users can only access their own analyses. Attempting to access another user's analysis will return 404.

---

### 6. Get All User Analyses
**Endpoint:** `GET /api/analyses`

**Authentication Required:** Yes (Bearer Token)

**Example Request:**
```javascript
const response = await fetch('http://localhost:8000/api/analyses', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});

const data = await response.json();
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "analyses": [
      {
        "id": 1,
        "file_name": "sample.mp4",
        "file_type": "video",
        "status": "completed",
        "created_at": "2025-12-11T10:30:00"
      },
      {
        "id": 2,
        "file_name": "audio.mp3",
        "file_type": "audio",
        "status": "processing",
        "created_at": "2025-12-11T11:00:00"
      }
    ],
    "count": 2
  },
  "error": null,
  "message": "Analyses retrieved successfully",
  "status_code": 200
}
```

**Possible Status Values:**
- `"processing"` - Analysis is in progress
- `"completed"` - Analysis finished successfully
- `"failed"` - Analysis encountered an error

---

## Authentication Flow

### Step 1: Register or Login
```javascript
// Register new user
const registerResponse = await fetch('http://localhost:8000/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'john_doe',
    password: 'password123'
  })
});

// Login
const loginResponse = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'john_doe',
    password: 'password123'
  })
});

const { data } = await loginResponse.json();
const accessToken = data.access_token; // Store this token
```

### Step 2: Store Token
Store the `access_token` securely:
- **LocalStorage** (simple but less secure)
- **SessionStorage** (cleared on tab close)
- **Memory/State** (cleared on refresh)
- **HttpOnly Cookie** (most secure, requires backend support)

```javascript
// Example with localStorage
localStorage.setItem('access_token', accessToken);
```

### Step 3: Use Token for Protected Endpoints
Include the token in all requests to protected endpoints:

```javascript
const token = localStorage.getItem('access_token');

const response = await fetch('http://localhost:8000/api/analyze', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
});
```

### Step 4: Handle Token Expiration
If you receive a 401 Unauthorized error, the token has expired. Redirect user to login:

```javascript
if (response.status === 401) {
  // Token expired - redirect to login
  localStorage.removeItem('access_token');
  window.location.href = '/login';
}
```

---

## Error Handling

All errors follow this format:
```json
{
  "success": false,
  "data": null,
  "error": "Error description",
  "message": null,
  "status_code": <HTTP_STATUS_CODE>
}
```

### Common HTTP Status Codes:
- **200**: Success
- **201**: Created (registration)
- **400**: Bad Request (validation error)
- **401**: Unauthorized (invalid/missing token)
- **404**: Not Found
- **413**: Payload Too Large (file > 20MB)
- **500**: Internal Server Error

---

## File Upload Specifications

### Supported File Types
- **Audio**: `.mp3`, `.wav`
- **Video**: `.mp4`, `.avi`

### File Size Limit
- Maximum: **20MB**

### Video Processing
When uploading video files:
1. Audio is automatically extracted from the video
2. The extracted audio is used for transcription and analysis
3. File type in response will be `"video"`

---

## CORS Configuration

The backend is configured to accept requests from:
```
FRONTEND_URL (from .env file)
```

Make sure your frontend URL matches the configured CORS origin.

---

## Example: Complete Upload Flow

```javascript
// 1. Login
const loginResponse = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'john', password: 'pass123' })
});

const { data: { access_token } } = await loginResponse.json();

// 2. Upload file for analysis
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const analyzeResponse = await fetch('http://localhost:8000/api/analyze', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${access_token}` },
  body: formData
});

const analysisResult = await analyzeResponse.json();

if (analysisResult.success) {
  console.log('Analysis ID:', analysisResult.data.analysis_id);
  console.log('Transcript:', analysisResult.data.transcript);
  console.log('WPM Data:', analysisResult.data.wpm_data);
}

// 3. Get all user analyses
const allAnalysesResponse = await fetch('http://localhost:8000/api/analyses', {
  method: 'GET',
  headers: { 'Authorization': `Bearer ${access_token}` }
});

const { data: { analyses, count } } = await allAnalysesResponse.json();
console.log(`Found ${count} analyses`);
```

---

## Notes for Frontend Developer

1. **Always use `/auth/login` for authentication** (not `/auth/token`)

2. **Token format:** Always include `Bearer ` prefix before the token:
   ```
   Authorization: Bearer <token>
   ```

3. **File uploads:** Use `FormData` and let the browser set the `Content-Type` header automatically

4. **Token storage:** Choose appropriate storage method based on security requirements

5. **Error handling:** Check `response.status` and `data.success` to handle errors properly

6. **Analysis processing time:** 
   - Small files (< 1 min): ~10-20 seconds
   - Medium files (1-5 min): ~30-60 seconds
   - Large files: May take several minutes

7. **WPM intervals:** 
   - Default: 2-second intervals
   - Each interval shows word count and words per minute

8. **Filler words detected:**
   - Single words: "um", "uh", "like", "you know", etc.
   - Multi-word phrases: "you know", "I mean", "sort of", etc.

9. **Loudness metrics:**
   - **RMS dB**: Root Mean Square loudness in decibels
   - **LUFS**: Loudness Units relative to Full Scale (broadcast standard)
   - Both measured in 1-second intervals

10. **Status polling:** If implementing real-time updates, poll `/api/analyze/{id}` until status changes from "processing" to "completed" or "failed"
