# sdsj_contest

Our solution for Sberbank Data Science Journey 2018 (https://sdsj.sberbank.ai/ru/contest). Unfortunately, we didn't manage to pass the last public test on memory, therefore we ended up on 144th place out of 224 with quality 4,20244 on private.

### How public was constructed (probably):
https://github.com/ilyenkov/sdsj-2018


### Ideas

##### Preprocessing

- clever .read_csv (for test only)
- parse datetime features
- downcast types
- drop irrelevant columns
- divide features by types (numeric/categorical)
- impute missing values
- filter used columns (for train only)

##### Model

- Catboost => No need to encode categorical string features
- flexible timing for training
- if dataset is too big, use more strict timing for training
