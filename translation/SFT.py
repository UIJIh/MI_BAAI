"""
12.19: cuda
01.03:
"""
#python SFT.py --model llama70 --data idioms
# $ nohup python -u .py & 
# $ tail -f nohup.out 
# watch -n 1 nvidia-smi
# wandb.login()
# wandb.init(
#     project='',
#     config={'method':''}
# )

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, random, wandb, argparse, re
torch.cuda.empty_cache()
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--data', required=True)
args = parser.parse_args()

if args.model == 'llama3':
    #model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "/share/project/huggingface/models/Meta-Llama-3.1-8B-Instruct"
elif args.model == 'llama70':
    #model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    model_id = "/share/project/huggingface/models/Meta-Llama-3.1-70B-Instruct"
elif args.model == 'mistral':
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
elif args.model == 'qwen2':
    model_id = "Qwen/Qwen2.5-7B-Instruct"

elif args.model == 'olmo':
    model_id = "allenai/OLMo-7B-0724-Instruct-hf"
elif args.model == 'post':
    model_id = "/data/uijih/8b_instruct/model_output/llama3_sft_idioms-NEW-2"

HF_token = '###' 

if args.model == 'llama70':
    # QLoRA 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Apply QLoRA
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    #tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"})
    #model.resize_token_embeddings(len(tokenizer))
    #model.print_trainable_parameters()
    model.config.use_cache = False

else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_token,
        trust_remote_code=True
    )
    # 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"}) # 안했더니 0.7에서 안떨어지는데 비교해보기
    model.resize_token_embeddings(len(tokenizer))

# 데이터셋 로드
if args.model == 'post':
    dataset_path = 'idioms_translation_conversations.jsonl' 
    with open(dataset_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
else:
    with open('/share/project/gsai/uiji/translation/idioms_conversations.jsonl', 'r') as f:
        raw_data = [json.loads(line) for line in f]

# 데이터셋을 텍스트 형식으로 변환
data = {'text': []}
for item in tqdm(raw_data):
    conversation = []
    for entry in item["conversations"]:
        if 'system' in entry:
            conversation.append({'role': 'system', 'content': entry['system']})  
        conversation.append({'role': 'user', 'content': entry['user']})
        conversation.append({'role': 'assistant', 'content': entry['assistant']})    
    # 대화 템플릿 적용 및 토큰화
    templated = tokenizer.apply_chat_template(conversation, tokenize=False, padding=True, max_length=200, truncation=True)
    data['text'].append(templated)

# 데이터 처리 완료 후 확인
print(f"Processed {len(data['text'])} conversations.")


# 데이터셋 변환
data = Dataset.from_dict(data)
shuffled_data = data.shuffle(seed=42)
print(f"Sample: \n{shuffled_data[0]}")

"""
데이터 길이 확인...
"""
# from collections import Counter
# lengths = [len(text.split()) for text in data['text']]
# print(f"평균 길이: {sum(lengths)/len(lengths)}, 최대 길이: {max(lengths)}")
# all_words = " ".join(data['text']).split()
# word_counts = Counter(all_words)
# print(word_counts.most_common(10))
# # 길이가 100 이상인 데이터 확인
# long_texts = [text for text in data['text'] if len(text.split()) > 100]
# print(f"긴 데이터 개수: {len(long_texts)}")
# print("샘플 데이터:", long_texts[:3])

# post
if args.model == 'post':
    training_args = SFTConfig(
        output_dir="./model_output/{}_sft_{}-NEW-2-post".format(args.model, args.data),
        dataset_text_field='text',
        remove_unused_columns=False,
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=4, 
        logging_steps=1,
        learning_rate=6e-5, 
        lr_scheduler_type="cosine", 
        packing=True,
        num_train_epochs=1, 
        save_strategy='no'  
    )
# epoch2 0.4356
# epoch1 0.4496

# original
else: 
        training_args = SFTConfig(
        output_dir="./model_output/{}_sft_{}-NEW-2-post".format(args.model, args.data),
        dataset_text_field='text',
        remove_unused_columns=False,
        per_device_train_batch_size=4, # max 200일때 32, max 늘리고선 둘이 바꿔버림 / NEW: 2->4
        gradient_accumulation_steps=64, # max 200일때 8
        logging_steps=1,
        learning_rate=2e-5, # NEW 1e-5 -> 2e-5
        lr_scheduler_type="cosine", # NEW
        #min_learning_rate=1e-7,  # NEW (최소 learning rate 설정) -> typeerror
        packing=True,
        #num_train_epochs=6, # 원래 세팅 3이었는데 NEW 너무 안내려가서 5로 올림 -> NEW-2에선 7 -> NEW-3에선 6
        num_train_epochs=3,# 70b
        save_strategy='no'  # 체크포인트 저장 비활성화
    )
    
sft_trainer = SFTTrainer(
    model,
    train_dataset=shuffled_data,
    args=training_args,
    #max_seq_length=512 peft none
)

print(f"Vocab before resize: {model.config.vocab_size}")
model.resize_token_embeddings(len(tokenizer))
print(f"Vocab after resize: {model.config.vocab_size}")

# TRAIN!!!
sft_trainer.train()

# save
tokenizer.save_pretrained(training_args.output_dir) # 안그러면 size mismatch
sft_trainer.save_model(training_args.output_dir)
print(f"!!! SAVED TO '{training_args.output_dir}' !!!")