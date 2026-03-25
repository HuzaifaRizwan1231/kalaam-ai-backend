import os
import asyncio
import sys
from unittest.mock import MagicMock
from fastapi import UploadFile
from src.controllers.analysis import AnalysisController
from src.entities.user import User
from src.config.db import SessionLocal, Base, engine

async def test_apple_video():
    # 0. Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    # 1. Mock a user
    db = SessionLocal()
    # Create test user if it doesn't exist
    test_user = db.query(User).filter(User.username == "testuser").first()
    if not test_user:
        test_user = User(username="testuser", hashed_password="fake")
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
    
    mock_user = test_user
    
    # 3. Path to the video
    video_path = r"c:\Users\iuiuy\Desktop\kalaam-ai\apple.mp4"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # 4. Open the file as a FastAPI UploadFile
    with open(video_path, "rb") as f:
        upload_file = UploadFile(
            filename="apple.mp4",
            file=f,
            headers={"content-type": "video/mp4"}
        )
        
        # 5. Initialize controller
        controller = AnalysisController()
        
        print(f"Starting analysis for {video_path}...")
        start_time = asyncio.get_event_loop().time()
        
        # 6. Run analysis
        try:
            response = await controller.create_analysis(upload_file, mock_user, db)
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            print(f"\nAnalysis completed in {duration:.2f} seconds")
            print(f"Status Code: {response.get('status_code')}")
            
            if response.get('status_code') == 200:
                data = response.get('data', {})
                print(f"Transcript: {data.get('transcript')[:100]}...")
                print(f"WPM: {data.get('wpm_data', {}).get('average_wpm')}")
                print(f"Head Direction: {data.get('head_direction_analysis') is not None}")
                print(f"Intonation: {data.get('intonation_analysis')}")
            else:
                print(f"Error: {response.get('message')}")
                
        except Exception as e:
            print(f"Exception during analysis: {str(e)}")
        finally:
            db.close()

if __name__ == "__main__":
    # Ensure src is in path
    sys.path.append(os.getcwd())
    asyncio.run(test_apple_video())
