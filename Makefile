.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y SignFlow || :
	@pip install -e .

# Example usage: 'make record_videos word="your_word" video_duration=10 num_videos=5'
.PHONY: record_videos
record_videos:
	python -c "from backend.ml_logic.registry import record_videos; record_videos($(word), $(video_duration), $(num_videos))"

.PHONY: run_main
run_main:
	cd backend/interface && python main.py
