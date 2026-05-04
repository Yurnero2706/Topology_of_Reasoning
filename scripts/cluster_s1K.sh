MODEL_NAME="Qwen/Qwen2.5-3B"
BLOCK_SIZE=32768 # Important: Set block size to 32768 to capture long-range dependencies in reasoning tasks [1]
LR=1e-5
EPOCHS=5
SAVE_STEPS=200 # Save checkpoint every 200 steps (adjust based on dataset size and training time) [2]

# --- Training loop for both datasets ---
DATASETS=("simplescaling/s1K" "simplescaling/s1K-1.1")

for DS in "${DATASETS[@]}"; do
  # Determine suffix based on dataset version for checkpoint naming
  if [[ "$DS" == *"1.1"* ]]; then
    SUFFIX="v1.1"
  else
    SUFFIX="v1.0"
  fi
  OUTPUT_DIR="checkpoints/qwen3b-s1-$SUFFIX"

  echo ">>> Bắt đầu huấn luyện SFT với $DS..."
  
  # Run torchrun for distributed training (adjust --nproc_per_node based on your GPU setup)
  torchrun --nproc_per_node=2 src/sft.py \
      --model_name=${MODEL_NAME} \
      --dataset_name=${DS} \
      --num_train_epochs=$EPOCHS \
      --learning_rate=$LR \
      --lr_scheduler_type="cosine" \
      --warmup_ratio=0.05 \
      --weight_decay=1e-4 \
      --adam_beta1=0.9 \
      --adam_beta2=0.95 \
      --per_device_train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --bf16=True \
      --gradient_checkpointing=True \
      --block_size=$BLOCK_SIZE \
      --fsdp="full_shard auto_wrap" \
      --output_dir=$OUTPUT_DIR \
      --save_steps=$SAVE_STEPS \
      --logging_steps=10
done

# --- Bước trích xuất và phân tích đồ thị để tạo Hình 9 ---
# Sau khi huấn luyện, bạn cần trích xuất trạng thái ẩn và tính đường kính
# Hình 9 so sánh qua các Layer Ratio: 0.1, 0.3, 0.5, 0.7, 0.9 [1]

LAYER_RATIOS=("0.1" "0.3" "0.5" "0.7" "0.9")
STEPS=("200" "400")

echo ">>> Bắt đầu trích xuất đồ thị tư duy và tính toán đường kính..."

for step in "${STEPS[@]}"; do
  for layer in "${LAYER_RATIOS[@]}"; do
    for suffix in "v1.0" "v1.1"; do
      MODEL_PATH="checkpoints/qwen3b-s1-$suffix/checkpoint-$step"
      
      # Lệnh trích xuất (Sử dụng mã nguồn từ repo gouki510/Topology_of_Reasoning)
      python src/extract_and_analyze.py \
          --model_path $MODEL_PATH \
          --dataset "AIME2024" \
          --target_layer_ratio $layer \
          --k_clusters 200 \
          --output_dir "results_fig9/step$step/$suffix/ratio$layer"
    done
  done
done

echo ">>> Completed all training and analysis steps. Check results_fig9/ for the extracted graphs and diameter calculations."
