# StreamingBench and OVO-Bench Raw Format Examples

这个文件只放本地原始标注中的真实样例，不用省略号替代题目、选项或关键字段。

## StreamingBench
### questions_real.json
```json
{
  "time": "[0:00:00 - 0:00:10]",
  "questions": [
    {
      "task_type": "Clips Summarize",
      "question": "Which of the following options best summarizes the actions of the individual taken just now?",
      "time_stamp": "00:00:31",
      "answer": "C",
      "options": [
        "A. The individual organized serving trays and sanitized the preparation area.",
        "B. The individual retrieved produce from the refrigerator and began chopping vegetables.",
        "C. The individual replenished bread inventory by placing new buns onto the shelves.",
        "D. The individual checked inventory levels and noted items lacking in stock."
      ],
      "required_ability": "episodic memory"
    }
  ],
  "video_categories": "preparation_of_meals",
  "video_path": "./videos/sample_348_real.mp4"
}
```

### questions_omni.json
```json
{
  "time": "[0:00:00 - 0:01:00]",
  "questions": [
    {
      "task_type": "Misleading Context Understanding",
      "question": "What's on the green table now?",
      "time_stamp": "00:00:50",
      "answer": "B",
      "options": [
        "A. There is a pink card in the middle of the table surrounded by four pairs of hands of different colours.",
        "B. There are four rows of playing cards on the table, three of which have pink faces.",
        "C. There is a white curtain in the middle of the table with a black vertical line in the middle that reads '590' on the left and '540' on the right..",
        "D. There is a white curtain in the middle of the table with two black lines on it."
      ],
      "required_ability": "working memory"
    }
  ],
  "video_categories": "online_tutorials",
  "video_path": "./videos/sample_2_Misleading_Context_Understanding.mp4"
}
```

### questions_sqa.json
```json
[
  {
    "time": "[0:01:00 - 0:02:00]",
    "questions": [
      {
        "task_type": "Sequential Question Answering",
        "question": "Who is preparing to serve the ball now?",
        "time_stamp": "00:00:36",
        "answer": "A",
        "options": [
          "Wang chuqin.",
          "Fan Z.D.",
          "Xu xin.",
          "Ma Lin."
        ],
        "required_ability": "episodic memory"
      }
    ],
    "video_categories": "table_tennis",
    "video_path": "./videos/sample_17_sqa.mp4"
  },
  {
    "time": "[0:02:00 - 0:03:00]",
    "questions": [
      {
        "task_type": "Sequential Question Answering",
        "question": "How much game score has the person mentioned in the last question scored now?",
        "time_stamp": "00:03:02",
        "answer": "C",
        "options": [
          "2.",
          "6.",
          "7.",
          "4."
        ],
        "required_ability": "episodic memory"
      }
    ],
    "video_categories": "table_tennis",
    "video_path": "./videos/sample_17_sqa.mp4"
  },
  {
    "time": "[0:03:00 - 0:04:00]",
    "questions": [
      {
        "task_type": "Sequential Question Answering",
        "question": "How much game score has the opponent of the person mentioned in the previous question scored now?",
        "time_stamp": "00:04:58",
        "answer": "A",
        "options": [
          "8.",
          "7.",
          "5.",
          "9."
        ],
        "required_ability": "episodic memory"
      }
    ],
    "video_categories": "table_tennis",
    "video_path": "./videos/sample_17_sqa.mp4"
  },
  {
    "time": "[0:03:00 - 0:04:00]",
    "questions": [
      {
        "task_type": "Sequential Question Answering",
        "question": "How much game score does the person mentioned in the first question have now?",
        "time_stamp": "00:08:07",
        "answer": "B",
        "options": [
          "4.",
          "3.",
          "8.",
          "2."
        ],
        "required_ability": "episodic memory"
      }
    ],
    "video_categories": "table_tennis",
    "video_path": "./videos/sample_17_sqa.mp4"
  },
  {
    "time": "[0:06:00 - 0:07:00]",
    "questions": [
      {
        "task_type": "Sequential Question Answering",
        "question": "How much game score has the person mentioned in the third question scored now?",
        "time_stamp": "00:09:57",
        "answer": "A",
        "options": [
          "4.",
          "2.",
          "8.",
          "6."
        ],
        "required_ability": "episodic memory"
      }
    ],
    "video_categories": "table_tennis",
    "video_path": "./videos/sample_17_sqa.mp4"
  }
]
```

### questions_proactive.json
```json
{
  "time": "[0:00:00 - 0:00:30]",
  "questions": [
    {
      "task_type": "Proactive Output",
      "question": "When the scoreboard shows 3 points for USA, output \"3\".",
      "time_stamp": "00:00:20",
      "ground_truth_time_stamp": "00:00:29",
      "ground_truth_output": "3",
      "required_ability": "working memory"
    }
  ],
  "video_categories": "basketball",
  "video_path": "./videos/sample_17_proactive.mp4"
}
```

### questions_real_stream.json
```json
{
  "questions": [
    {
      "task_type": "Object Recognition",
      "question": "What can be seen in the cockpit's center screen right now?",
      "time_stamp": "00:00:06",
      "answer": "B",
      "options": [
        "A. A compass.",
        "B. A map.",
        "C. An altitude meter.",
        "D. A weather radar."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/real/sample_416_00_00_06.wav",
      "time": "[0:00:00 - 0:00:20]"
    },
    {
      "task_type": "Attribute Recognition",
      "question": "What colors are the pilot's handheld controls right now?",
      "time_stamp": "00:02:37",
      "answer": "A",
      "options": [
        "A. Black.",
        "B. Red and green.",
        "C. Yellow and blue.",
        "D. Black and orange."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/real/sample_416_00_02_37.wav",
      "time": "[0:02:23 - 0:02:43]"
    },
    {
      "task_type": "Attribute Recognition",
      "question": "What are the visible weather conditions right now?",
      "time_stamp": "00:05:01",
      "answer": "A",
      "options": [
        "A. Clear sky with scattered clouds.",
        "B. Overcast sky with rain.",
        "C. Foggy with low visibility.",
        "D. Thunderstorms with lightning."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/real/sample_416_00_05_01.wav",
      "time": "[0:04:46 - 0:05:06]"
    },
    {
      "task_type": "Attribute Recognition",
      "question": "What are colors of the curved flight path right now?",
      "time_stamp": "00:07:15",
      "answer": "D",
      "options": [
        "A. Yellow and blue.",
        "B. Blue and white.",
        "C. Green and ograne.",
        "D. Red and blue."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/real/sample_416_00_07_15.wav",
      "time": "[0:07:09 - 0:07:29]"
    },
    {
      "task_type": "Object Recognition",
      "question": "What is located to the bottom-left of the cockpit's central control panel right now?",
      "time_stamp": "00:09:33",
      "answer": "A",
      "options": [
        "A. A blue handle.",
        "B. A red button.",
        "C. An altitude meter.",
        "D. A compass."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/real/sample_416_00_09_33.wav",
      "time": "[0:09:32 - 0:09:52]"
    }
  ],
  "video_path": "./videos/sample_416_real.mp4"
}
```

### questions_omni_stream.json
```json
{
  "questions": [
    {
      "task_type": "Misleading Context Understanding",
      "question": "What's on the green table now?",
      "time_stamp": "00:00:50",
      "answer": "B",
      "options": [
        "A. There is a pink card in the middle of the table surrounded by four pairs of hands of different colours.",
        "B. There are four rows of playing cards on the table, three of which have pink faces.",
        "C. There is a white curtain in the middle of the table with a black vertical line in the middle that reads '590' on the left and '540' on the right..",
        "D. There is a white curtain in the middle of the table with two black lines on it."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/omni/sample_2_Misleading_Context_Understanding_00_00_50.wav",
      "time": "[0:00:00 - 0:01:00]"
    },
    {
      "task_type": "Misleading Context Understanding",
      "question": "What exactly is on the table now?",
      "time_stamp": "00:01:52",
      "answer": "C",
      "options": [
        "A. There are five white circles on the table, from left to right, the circles contain the club symbol, the diamond symbol, the heart symbol, the spade symbol and 'NT'.",
        "B. There are five white circles on the table, but only the first circle on the left has the clubs symbol in it, the other circles have nothing in them.",
        "C. There is now a spade, a club, a heart and a diamond on the table, with a white circle in the centre that says 'NT'.",
        "D. There are five white circles on the table and nothing inside them."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/omni/sample_2_Misleading_Context_Understanding_00_01_52.wav",
      "time": "[0:01:00 - 0:02:00]"
    },
    {
      "task_type": "Misleading Context Understanding",
      "question": "How many copies of playing cards are on the table now?",
      "time_stamp": "00:03:45",
      "answer": "D",
      "options": [
        "A. There is now only one copy of playing cards on the table.",
        "B. There are now five copies of playing cards on the table.",
        "C. There are now only two copies of playing cards on the table.",
        "D. There are now thirteen copies of playing cards on the table."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/omni/sample_2_Misleading_Context_Understanding_00_03_45.wav",
      "time": "[0:03:00 - 0:04:00]"
    },
    {
      "task_type": "Misleading Context Understanding",
      "question": "What else exists on the green table now?",
      "time_stamp": "00:04:50",
      "answer": "A",
      "options": [
        "A. Now there is a white curtain on the table with three black lines and the letters '4NTX4', '130X4' and '30'.",
        "B. The table now has a white curtain with two black lines and the letters '4NTX4', '130X4' and '30'.",
        "C. Now there is a white curtain on the table with two black lines on it.",
        "D. There is a white circle in the middle of the table with 'NT' written inside."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/omni/sample_2_Misleading_Context_Understanding_00_04_50.wav",
      "time": "[0:04:00 - 0:05:00]"
    },
    {
      "task_type": "Misleading Context Understanding",
      "question": "What's inside the white curtain on the tabletop?",
      "time_stamp": "00:05:46",
      "answer": "B",
      "options": [
        "A. In the centre of the white cloth there is a black line with '1 30' and '2 60' written on the left side, and in the middle of the numbers there is a spade and a heart symbol, respectively, and below this set of numbers there is another black line and '590'.",
        "B. Inside the white curtain there is a vertical line with '590' written on the left and '540' written on the right with a red circle on the left.",
        "C. A white cloth with two black lines and '1 30' written in the lower left corner of the curtain with a spade in the middle of the numbers.",
        "D. A white cloth with two black lines crossed and '-200x4' written in the upper left corner of the curtain."
      ],
      "required_ability": "working memory",
      "audio_path": "./audio/omni/sample_2_Misleading_Context_Understanding_00_05_46.wav",
      "time": "[0:05:00 - 0:06:00]"
    }
  ],
  "video_path": "./videos/sample_2_Misleading_Context_Understanding.mp4"
}
```

### questions_sqa_stream.json
```json
{
  "questions": [
    {
      "task_type": "Sequential Question Answering",
      "question": "Who is preparing to serve the ball now?",
      "time_stamp": "00:00:36",
      "answer": "A",
      "options": [
        "Wang chuqin.",
        "Fan Z.D.",
        "Xu xin.",
        "Ma Lin."
      ],
      "required_ability": "episodic memory",
      "audio_path": "./audio/sqa/sample_17_00_00_36.wav",
      "time": "[0:01:00 - 0:02:00]"
    },
    {
      "task_type": "Sequential Question Answering",
      "question": "How much game score has the person mentioned in the last question scored now?",
      "time_stamp": "00:03:02",
      "answer": "C",
      "options": [
        "2.",
        "6.",
        "7.",
        "4."
      ],
      "required_ability": "episodic memory",
      "audio_path": "./audio/sqa/sample_17_00_03_02.wav",
      "time": "[0:02:00 - 0:03:00]"
    },
    {
      "task_type": "Sequential Question Answering",
      "question": "How much game score has the opponent of the person mentioned in the previous question scored now?",
      "time_stamp": "00:04:58",
      "answer": "A",
      "options": [
        "8.",
        "7.",
        "5.",
        "9."
      ],
      "required_ability": "episodic memory",
      "audio_path": "./audio/sqa/sample_17_00_04_58.wav",
      "time": "[0:03:00 - 0:04:00]"
    },
    {
      "task_type": "Sequential Question Answering",
      "question": "How much game score does the person mentioned in the first question have now?",
      "time_stamp": "00:08:07",
      "answer": "B",
      "options": [
        "4.",
        "3.",
        "8.",
        "2."
      ],
      "required_ability": "episodic memory",
      "audio_path": "./audio/sqa/sample_17_00_08_07.wav",
      "time": "[0:03:00 - 0:04:00]"
    },
    {
      "task_type": "Sequential Question Answering",
      "question": "How much game score has the person mentioned in the third question scored now?",
      "time_stamp": "00:09:57",
      "answer": "A",
      "options": [
        "4.",
        "2.",
        "8.",
        "6."
      ],
      "required_ability": "episodic memory",
      "audio_path": "./audio/sqa/sample_17_00_09_57.wav",
      "time": "[0:06:00 - 0:07:00]"
    }
  ],
  "video_categories": "table_tennis",
  "video_path": "./videos/sample_17_sqa.mp4"
}
```

## OVO-Bench
下面每个 task 先给原始 annotation，再给我们/官方会实际喂给模型的 prompt。REC、SSR、CRR 的 raw annotation 里不是都直接有 question 字段，所以这里额外展开生成后的 prompt。

### EPM
Raw annotation:
```json
{
  "id": 0,
  "task": "EPM",
  "video": "Ego4D/clips/ec4a3ba3-eb00-4aa8-9b41-36043ece98f7.mp4",
  "realtime": 215,
  "question": "Who did I communicate to  when chopping egg plants?",
  "answer": "a person with blue shirt",
  "options": [
    "a person with brown shirt",
    "a person with green shirt",
    "a person with blue shirt",
    "a person with white shirt"
  ],
  "gt": 2
}
```
Model prompt:
```text
Question: Who did I communicate to  when chopping egg plants?
Options:
A. a person with brown shirt; B. a person with green shirt; C. a person with blue shirt; D. a person with white shirt;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### ASI
Raw annotation:
```json
{
  "id": 483,
  "task": "ASI",
  "video": "COIN/Hf2AisK1wHY.mp4",
  "realtime": 136,
  "question": "What does the person do after load the wheel",
  "answer": "pump up the tire",
  "options": [
    "pump up the tire",
    "unload the wheel",
    "load the tire",
    "load the inner tube"
  ],
  "gt": 0
}
```
Model prompt:
```text
Question: What does the person do after load the wheel
Options:
A. pump up the tire; B. unload the wheel; C. load the tire; D. load the inner tube;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### HLD
Raw annotation:
```json
{
  "id": 297,
  "task": "HLD",
  "video": "Ego4D/clips/93231c7e-1cf4-4a20-b1f8-9cc9428915b2.mp4",
  "question": "what did I put in the black dustbin?",
  "answer": "Unable to answer",
  "options": [
    "empty water bottles",
    "Unable to answer",
    "old newspapers",
    "food scraps"
  ],
  "gt": 1,
  "realtime": 371
}
```
Model prompt:
```text
Question: what did I put in the black dustbin?
Options:
A. empty water bottles; B. Unable to answer; C. old newspapers; D. food scraps;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### OCR
Raw annotation:
```json
{
  "id": 1319,
  "task": "OCR",
  "video": "AutoEvalMetaData/106.mp4",
  "realtime": 54.2,
  "question": "What are the names listed on each lane of the pool?",
  "answer": "Rapsys, Di Cola, De Tullio, Popovici, Djakovic, Maertens, Auboek, Markos",
  "options": [
    "Rapsys, Di Cola, De Tullio, Popovici, Djakovic, Maertens, Auboek, Markos",
    "Leclos, Vekovishchev, Seto, Irie, Grothe, Schooling, Solaeche, Lazor",
    "Fraser-Holmes, Krutohzmov, Meyer, Hosszu, Kromowidjojo, Stupp, Ohashi, King",
    "Lochte, Dressel, Phelps, Cseh, Chalmers, Larkin, Peaty, Kenderesi"
  ],
  "gt": 0
}
```
Model prompt:
```text
Question: What are the names listed on each lane of the pool?
Options:
A. Rapsys, Di Cola, De Tullio, Popovici, Djakovic, Maertens, Auboek, Markos; B. Leclos, Vekovishchev, Seto, Irie, Grothe, Schooling, Solaeche, Lazor; C. Fraser-Holmes, Krutohzmov, Meyer, Hosszu, Kromowidjojo, Stupp, Ohashi, King; D. Lochte, Dressel, Phelps, Cseh, Chalmers, Larkin, Peaty, Kenderesi;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### ACR
Raw annotation:
```json
{
  "id": 1210,
  "task": "ACR",
  "video": "Ego4D/video/e1fa5de2-eb21-41a3-85b3-882a3652df42.mp4",
  "realtime": 201.93,
  "question": "What is he doing with that tool?",
  "answer": "He is inflating the tire with the tool.",
  "options": [
    "He is repairing a bicycle.",
    "She is using a jack to lift a car.",
    "He is tightening bolts on a wheel.",
    "He is inflating a tire."
  ],
  "gt": 3
}
```
Model prompt:
```text
Question: What is he doing with that tool?
Options:
A. He is repairing a bicycle.; B. She is using a jack to lift a car.; C. He is tightening bolts on a wheel.; D. He is inflating a tire.;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### ATR
Raw annotation:
```json
{
  "id": 993,
  "task": "ATR",
  "video": "YouTube_Games/PLJ3VIGhVd3r8Int6IZT_v3S_BzG9RVfiG&index=1.mp4",
  "realtime": 279,
  "question": "What is the color of the dog?",
  "answer": "Black.",
  "options": [
    "White.",
    "Black.",
    "Pink.",
    "Brown."
  ],
  "gt": 1
}
```
Model prompt:
```text
Question: What is the color of the dog?
Options:
A. White.; B. Black.; C. Pink.; D. Brown.;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### STU
Raw annotation:
```json
{
  "id": 631,
  "task": "STU",
  "video": "YouTube_Games/PLJ3VIGhVd3r8Int6IZT_v3S_BzG9RVfiG&index=1.mp4",
  "realtime": 172,
  "question": "Where is the biggest giant standing?",
  "answer": "In the back of other four giants.",
  "options": [
    "In the front of other four giants.",
    "In the back of other four giants.",
    "In the left of other four giants.",
    "In the right of other four giants."
  ],
  "gt": 1
}
```
Model prompt:
```text
Question: Where is the biggest giant standing?
Options:
A. In the front of other four giants.; B. In the back of other four giants.; C. In the left of other four giants.; D. In the right of other four giants.;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### FPD
Raw annotation:
```json
{
  "id": 1109,
  "task": "FPD",
  "video": "Ego4D/video/005efbab-c314-48d8-8dd9-bfc1263ec5df.mp4",
  "realtime": 88.43,
  "question": "What is this person going to do?",
  "answer": "The person is going to drill a hole into the wires or material in front of them.",
  "options": [
    "The individual intends to plant a tree.",
    "The person is going to drill a hole.",
    "The person is planning to paint a mural.",
    "The person is preparing to change a tire."
  ],
  "gt": 1
}
```
Model prompt:
```text
Question: What is this person going to do?
Options:
A. The individual intends to plant a tree.; B. The person is going to drill a hole.; C. The person is planning to paint a mural.; D. The person is preparing to change a tire.;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### OJR
Raw annotation:
```json
{
  "id": 809,
  "task": "OJR",
  "video": "YouTube_Games/PLJ3VIGhVd3r8Int6IZT_v3S_BzG9RVfiG&index=1.mp4",
  "realtime": 172,
  "question": "What is under the monkey?",
  "answer": "A piece of cloud.",
  "options": [
    "The trees",
    "A piece of rock.",
    "A piece of cloud.",
    "The ground."
  ],
  "gt": 2
}
```
Model prompt:
```text
Question: What is under the monkey?
Options:
A. The trees; B. A piece of rock.; C. A piece of cloud.; D. The ground.;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

### REC
Raw annotation:
```json
{
  "id": 1558,
  "task": "REC",
  "video": "perception_test/video_4017.mp4",
  "activity": "breaking something",
  "start_times": [
    6
  ],
  "end_times": [
    17
  ],
  "test_info": [
    {
      "realtime": 17,
      "count": 1
    },
    {
      "realtime": 19,
      "count": 1
    },
    {
      "realtime": 32.0,
      "count": 1
    }
  ]
}
```
Model prompt:
```text
You're watching a video in which people may perform a certain type of action repetitively.
The person performing this kind of action is referred to as "they" in the following statement.
Your task is to count how many times different people in the video have performed this kind of action in total.
One complete motion counts as one.
Now, answer the following question: How many times did they breaking something?
Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.
Do not include any additional text or explanation in your response.
```

### SSR
Raw annotation:
```json
{
  "id": 1516,
  "task": "SSR",
  "video": "COIN/xZecGPPhbHE.mp4",
  "tutorial": "PutOnHairExtensions",
  "all_steps": [
    "pull up the hair to reserve place for the hair extensions",
    "put on the hair extensions",
    "put down the hair and comb"
  ],
  "start_time": [
    25,
    31,
    107
  ],
  "end_time": [
    30,
    49,
    117
  ],
  "test_info": [
    {
      "realtime": 27,
      "step": "pull up the hair to reserve place for the hair extensions",
      "type": 1
    },
    {
      "realtime": 30,
      "step": "pull up the hair to reserve place for the hair extensions",
      "type": 1
    },
    {
      "realtime": 26,
      "step": "put on the hair extensions",
      "type": 0
    },
    {
      "realtime": 29,
      "step": "put on the hair extensions",
      "type": 0
    },
    {
      "realtime": 33,
      "step": "put on the hair extensions",
      "type": 1
    },
    {
      "realtime": 36,
      "step": "put on the hair extensions",
      "type": 1
    },
    {
      "realtime": 49,
      "step": "put on the hair extensions",
      "type": 1
    },
    {
      "realtime": 102,
      "step": "put down the hair and comb",
      "type": 0
    },
    {
      "realtime": 105,
      "step": "put down the hair and comb",
      "type": 0
    },
    {
      "realtime": 109,
      "step": "put down the hair and comb",
      "type": 1
    },
    {
      "realtime": 112,
      "step": "put down the hair and comb",
      "type": 1
    },
    {
      "realtime": 117,
      "step": "put down the hair and comb",
      "type": 1
    }
  ]
}
```
Model prompt:
```text
You're watching a tutorial video which contains a sequence of steps.
The following is one step from the whole procedure:
pull up the hair to reserve place for the hair extensions
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
```

### CRR
Raw annotation:
```json
{
  "id": 1468,
  "task": "CRR",
  "video": "MovieNet/tt0048028.mp4",
  "question": "The woman in a black coat walks towards the direction of the black car, what action does she take to the car?",
  "answer": "She walks past the car.",
  "ask_time": 297,
  "clue_time": 308,
  "test_info": [
    {
      "realtime": 297,
      "type": 0
    },
    {
      "realtime": 302,
      "type": 0
    },
    {
      "realtime": 310,
      "type": 1
    },
    {
      "realtime": 318,
      "type": 1
    },
    {
      "realtime": 338,
      "type": 1
    }
  ]
}
```
Model prompt:
```text
You're responsible for answering questions based on the video content.
The following question is relevant to the latest frames, i.e. the end of the video.
The woman in a black coat walks towards the direction of the black car, what action does she take to the car?
Decide whether the existing visual content, especially the latest frames near the end of the video, provides enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
```

