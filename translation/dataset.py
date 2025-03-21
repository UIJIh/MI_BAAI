import pandas as pd
import json, random

"""
영어 데이터셋 크기: 3849
한국어 데이터셋 크기: 3989
"""

random.seed(42)

en_idiom_file_path = '/share/project/gsai/uiji/datasets/filtered_EN_idiom_3849.csv'
kr_idiom_file_path = '/share/project/gsai/uiji/datasets/KR_idiom_3989.csv'
en_data = pd.read_csv(en_idiom_file_path)
kr_data = pd.read_csv(kr_idiom_file_path)

# def clean_data(df, idiom_col, meaning_col):
#     # 중복 제거
#     df = df.drop_duplicates(subset=[idiom_col])
#     # 결측치 제거
#     df = df.dropna(subset=[idiom_col, meaning_col])
#     # 인덱스 재설정
#     df = df.reset_index(drop=True)
#     return df
# en_data = clean_data(en_data, idiom_col='idiom', meaning_col='en_meaning')
# kr_data = clean_data(kr_data, idiom_col='Idiom', meaning_col='Meaning')

# 데이터셋 크기 확인
print(f"영어 데이터셋 크기: {len(en_data)}")
print(f"한국어 데이터셋 크기: {len(kr_data)}")

# 데이터셋 크기를 작은 쪽에 맞추기 (랜덤 샘플링)
# min_size = min(len(en_data), len(kr_data))
# en_data = en_data.sample(n=min_size, random_state=42).reset_index(drop=True)
# kr_data = kr_data.sample(n=min_size, random_state=42).reset_index(drop=True)
# print(f"영어 데이터셋 크기: {len(en_data)}")
# print(f"한국어 데이터셋 크기: {len(kr_data)}")

conversations_dataset = []

# templates
# 8 
idiom_to_meaning_templates_en = [
    "Can you explain the meaning of the idiom '{idiom}'?",
    "What does the idiom '{idiom}' signify?",
    "I came across the idiom '{idiom}'. Could you explain what it means?",
    "In simple words, what does the idiom '{idiom}' mean?",
    "Could you help me understand the idiom '{idiom}'?",
    "What is the meaning of '{idiom}' in everyday usage?",
    "If someone says '{idiom}', what does it imply?",
    "What message is conveyed by the idiom '{idiom}'?"
]
# 8
meaning_to_idiom_templates_en = [
    "Could you provide an idiom that matches the meaning '{meaning}'?",
    "What idiom represents the meaning '{meaning}'? Answer in one line.",
    "If I want to express '{meaning}', what idiom can I use?",
    "Please tell me an idiom that fits the description: '{meaning}'.",
    "Is there an idiom that captures the meaning '{meaning}'?",
    "How can I describe '{meaning}' using an idiom?",
    "Which idiom best describes the meaning '{meaning}'?",
    "Can you suggest an idiom for '{meaning}'?"
]
# 8
idiom_to_meaning_templates_kr = [
    "관용구 '{idiom}'의 의미를 설명해 주시겠어요?",
    "관용구 '{idiom}'은 무엇을 뜻하나요?",
    "'{idiom}'이라는 관용구를 봤는데, 무슨 뜻인지 알려주시겠어요?",
    "'{idiom}'은 어떤 의미인가요?",
    "'{idiom}'이라는 표현은 무슨 의미인가요?",
    "'{idiom}'라는 관용구는 어떤 의미로 사용되나요?",
    "'{idiom}'라는 관용 표현은 어떤 의미인가요?",
    "'{idiom}'이라는 관용구의 뜻을 간단히 알려주실 수 있나요?"
]
# 9
meaning_to_idiom_templates_kr = [
    "'{meaning}'라는 의미를 가진 관용 표현이 있을까요?",
    "어떤 관용구가 '{meaning}'을 잘 나타낼까요?",
    "'{meaning}'에 딱 맞는 관용구를 알려주실 수 있나요?",
    "'{meaning}'을 표현할 수 있는 관용 표현은 무엇인가요?",
    "'{meaning}'와 비슷한 뜻을 가진 관용표현이 있을까요?",
    "'{meaning}'을 가장 잘 설명할 수 있는 관용구는 무엇일까요?",
    "'{meaning}'을 말하고 싶을 때 사용할 수 있는 관용구가 있을까요?",
    "'{meaning}'을 표현하려면 어떤 관용표현이 적당할까요?",
    "관용구 중에서 '{meaning}'이라는 뜻을 가진 것이 있을까요?"
    ]

# 어시스턴트 템플릿
# 6
assistant_templates_idiom_to_meaning_en = [
    "The idiom means: {meaning}",
    "'{idiom}' means: {meaning}",
    "The meaning of '{idiom}' is: {meaning}",
    "'{idiom}' signifies: {meaning}",
    "'{idiom}' carries the meaning: {meaning}.",    
    "{meaning}"
]
# 6
assistant_templates_meaning_to_idiom_en = [
    "An idiom that matches this meaning is: '{idiom}'",
    "An appropriate idiom would be: '{idiom}'",
    "'{idiom}' perfectly captures the idea.",
    "A fitting idiom would be: '{idiom}'",
    "To express the meaning '{meaning}', you can use the idiom '{idiom}'.",
    "{idiom}"
]
# 6
assistant_templates_idiom_to_meaning_kr = [
    "'{idiom}'의 의미는 다음과 같습니다: {meaning}",
    "'{meaning}'을 뜻합니다.",
    "'{idiom}'는 '{meaning}'이라는 의미입니다.",
    "해당 관용구의 뜻은 다음과 같습니다: {meaning}",
    "'{idiom}'은 주로 '{meaning}'이라는 의미로 사용됩니다.",
    "{meaning}"
]
# 7
assistant_templates_meaning_to_idiom_kr = [
    "이 의미에 맞는 한국어 관용구는 '{idiom}'입니다.",
    "'{idiom}'이라는 관용구를 사용할 수 있습니다.",
    "해당 의미를 가진 관용구는 '{idiom}'입니다.",
    "'{meaning}'을 표현하기에 '{idiom}'이 적합한 관용구입니다.",
    "'{idiom}'가 적합한 관용표현입니다.",
    "'{idiom}'라는 관용구가 이 의미를 담고 있습니다.",
    "{idiom}"
]

# 영어 데이터 처리
for index in range(len(en_data)):
    en_row = en_data.iloc[index]
    # Idiom-to-Meaning (영어)
    user_template = random.choice(idiom_to_meaning_templates_en)
    assistant_template = random.choice(assistant_templates_idiom_to_meaning_en)
    conversation_en_idiom_to_meaning = {
        "conversations": [
            {
                "user": user_template.format(idiom=en_row['idiom']),
                "assistant": assistant_template.format(idiom=en_row['idiom'], meaning=en_row['en_meaning'])
            }
        ]
    }
    conversations_dataset.append(conversation_en_idiom_to_meaning)
    # Meaning-to-Idiom (영어)
    user_template = random.choice(meaning_to_idiom_templates_en)
    assistant_template = random.choice(assistant_templates_meaning_to_idiom_en)
    conversation_en_meaning_to_idiom = {
        "conversations": [
            {
                "user": user_template.format(meaning=en_row['en_meaning']),
                "assistant": assistant_template.format(idiom=en_row['idiom'], meaning=en_row['en_meaning'])
            }
        ]
    }
    conversations_dataset.append(conversation_en_meaning_to_idiom)

# 한국어 데이터 처리
for index in range(len(kr_data)):
    kr_row = kr_data.iloc[index]
    # Idiom-to-Meaning (한국어)
    user_template = random.choice(idiom_to_meaning_templates_kr)
    assistant_template = random.choice(assistant_templates_idiom_to_meaning_kr)
    conversation_kr_idiom_to_meaning = {
        "conversations": [
            {
                "user": user_template.format(idiom=kr_row['Idiom']),
                "assistant": assistant_template.format(idiom=kr_row['Idiom'], meaning=kr_row['Meaning'])
            }
        ]
    }
    conversations_dataset.append(conversation_kr_idiom_to_meaning)
    # Meaning-to-Idiom (한국어)
    user_template = random.choice(meaning_to_idiom_templates_kr)
    assistant_template = random.choice(assistant_templates_meaning_to_idiom_kr)
    conversation_kr_meaning_to_idiom = {
        "conversations": [
            {
                "user": user_template.format(meaning=kr_row['Meaning']),
                "assistant": assistant_template.format(idiom=kr_row['Idiom'], meaning=kr_row['Meaning'])
            }
        ]
    }
    conversations_dataset.append(conversation_kr_meaning_to_idiom)

random.shuffle(conversations_dataset)
output_file_path = 'idioms_conversations.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
    for data_entry in conversations_dataset:
        jsonl_file.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

print(f"({len(conversations_dataset)}) saved to '{output_file_path}'")
