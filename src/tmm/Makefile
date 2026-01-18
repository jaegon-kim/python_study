.PHONY: triton clean clean-all

triton:
	bash /root/Workspace/python_study/src/tmm/scripts/build_triton_venv.sh

clean:
	rm -f /root/Workspace/python_study/src/tmm/triton_model_repository.tar.gz

clean-all: clean
	bash /root/Workspace/python_study/src/tmm/scripts/build_triton_venv.sh --clean
	rm -rf /root/Workspace/python_study/src/tmm/triton_model_repository/ttm/1/ttm_model
