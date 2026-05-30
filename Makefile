.PHONY: help test train-single train-ddp train-fsdp comm-bench plot clean

help:  ## 显示帮助信息
	@echo "MiniTrainSys - 分布式训练系统与性能剖析框架"
	@echo ""
	@echo "用法:"
	@echo "  make test           运行单元测试"
	@echo "  make train-single   Single GPU 训练"
	@echo "  make train-ddp      DDP 2 GPU 训练"
	@echo "  make train-fsdp     FSDP 2 GPU 训练"
	@echo "  make comm-bench     通信算子 benchmark"
	@echo "  make plot           生成实验图表"
	@echo "  make clean          清理临时文件"
	@echo ""
	@echo "配置文件: configs/*.yaml"
	@echo "文档: docs/*.md"

test:  ## 运行单元测试
	pytest tests/ -v

train-single:  ## Single GPU 训练
	bash scripts/run_single.sh

train-ddp:  ## DDP 2 GPU 训练
	bash scripts/run_ddp_2gpu.sh

train-fsdp:  ## FSDP 2 GPU 训练
	bash scripts/run_fsdp_2gpu.sh

comm-bench:  ## 通信算子 benchmark
	bash scripts/run_comm_bench.sh

plot:  ## 生成实验图表
	python scripts/plot_results.py

clean:  ## 清理临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
