export MODEL_NAME="stabilityai/stable-diffusion-2"
export OUTPUT_DIR="output"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=400 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval=100000 \
  --save_sample_prompt="photo of zwx person" \
  --instance_prompt="photo of zwx person" \
  --class_prompt="photo of a person" \
  --instance_data_dir="./training" \
  --class_data_dir="./regularization/person"

export WEIGHTS_DIR=output/800
export ckpt_path=$WEIGHTS_DIR/model.ckpt

python convert_diffusers_to_original_stable_diffusion.py --model_path $WEIGHTS_DIR  --checkpoint_path $ckpt_path