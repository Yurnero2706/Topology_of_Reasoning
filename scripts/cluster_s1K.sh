# --- Cấu hình chung (Phụ lục J) ---
MODEL_NAME="Qwen/Qwen2.5-3B"
BLOCK_SIZE=32768
LR=1e-5
EPOCHS=5
SAVE_STEPS=200 # Để lưu checkpoint tại bước 200 và 400 [2]

# --- 1. Giai đoạn Huấn luyện SFT (v1.0 và v1.1) ---
DATASETS=("simplescaling/s1K" "simplescaling/s1K-1.1")

for DS in "${DATASETS[@]}"; do
  SUFFIX=$([[ "$DS" == *"1.1"* ]] && echo "v1.1" || echo "v1.0")
  OUTPUT_DIR="checkpoints/qwen3b-s1-$SUFFIX"

  echo ">>> Đang huấn luyện SFT với $DS..."
  # Effective batch size = 2 (GPUs) * 1 (per_device) * 4 (grad_accum) = 8 [3]
  torchrun --nproc_per_node=2 src/sft.py \
      --model_name_or_path=${MODEL_NAME} \
      --dataset_name=${DS} \
      --num_train_epochs=${EPOCHS} \
      --learning_rate=${LR} \
      --lr_scheduler_type="cosine" \
      --warmup_ratio=0.05 \
      --weight_decay=1e-4 \
      --adam_beta1=0.9 \
      --adam_beta2=0.95 \
      --per_device_train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --bf16=True \
      --gradient_checkpointing=True \
      --block_size=${BLOCK_SIZE} \
      --fsdp="full_shard auto_wrap" \
      --output_dir=${OUTPUT_DIR} \
      --save_steps=${SAVE_STEPS} \
      --logging_steps=10
done

# --- 2. Giai đoạn Phân tích Đồ thị (Sử dụng cluster_steps_generated.py) ---
# Trục X của Hình 9 là Layer Ratio: 0.1, 0.3, 0.5, 0.7, 0.9 [4, 5]
LAYER_RATIOS=("0.1" "0.3" "0.5" "0.7" "0.9")
STEPS=("200" "400")

echo ">>> Bắt đầu phân cụm và tính toán thuộc tính đồ thị..."

for step in "${STEPS[@]}"; do
  for layer in "${LAYER_RATIOS[@]}"; do
    for suffix in "v1.0" "v1.1"; do
      MODEL_PATH="checkpoints/qwen3b-s1-$suffix/checkpoint-$step"
      
      echo "Xử lý: $suffix | Step $step | Layer Ratio $layer"
      
      # cluster_steps_generated.py thực hiện K-means (K=200) và tính diameter [6, 7]
      python src/cluster_steps_generated.py \
          --model_name_or_path "$MODEL_PATH" \
          --dataset "simplescaling/s1K" \
          --target_layer_ratio "$layer" \
          --k_clusters 200 \
          --output_dir "extract_steps/$suffix/step$step/ratio$layer"
    done
  done
done

echo ">>> Hoàn thành. Kết quả results.json đã sẵn sàng để vẽ biểu đồ tương tự Hình 9."
