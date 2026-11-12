# Результаты оценки RAG-бота

- Дата: 2026-11-12 17:12
- Эмбеддинги: `intfloat/multilingual-e5-small`, чанки ≈500 симв., перекрытие 100, top-k = 5
- Генерация: `— (не запускалась)`; судья: `—`
- Вопросов: 30 с ответом (простые — 20, перефразированные — 2, два документа — 4, подвох в цифрах — 4) + 10 без ответа (нет в документах — 5, ловушки (похожие) — 5)

## Ключевые метрики

| Метрика | Значение |
|---|---|
| Retrieval hit@1 | 93% (28/30) |
| Retrieval hit@3 | 100% (30/30) |
| Retrieval MRR@5 | 0.961 |

## По категориям: вопросы с ответом

| Категория | N | hit@3 | all@5 |
|---|---|---|---|
| Простые | 20 | 100% (20/20) | 100% (20/20) |
| Перефразированные | 2 | 100% (2/2) | 100% (2/2) |
| Два документа | 4 | 100% (4/4) | 75% (3/4) |
| Подвох в цифрах | 4 | 100% (4/4) | 100% (4/4) |
| **Все с ответом** | 30 | 100% (30/30) | 97% (29/30) |

`all@5` — все ожидаемые документы попали в top-5 (контекст LLM); важно для вопросов на два документа.

## Вопросы с ответом — подробно

| # | Категория | Вопрос | Ожид. документы | top-3 | hit@3 |
|---|---|---|---|---|---|
| 1 | Простые | Сколько всего календарных дней оплачиваемого отпуска в год получает с… | 01_otpuska.md | 01_otpuska.md, 01_otpuska.md, 01_otpuska.md | ✅ |
| 2 | Простые | За сколько дней нужно подать заявление на отпуск? | 01_otpuska.md | 01_otpuska.md, 01_otpuska.md, 01_otpuska.md | ✅ |
| 3 | Простые | Сколько неиспользованных дней отпуска можно перенести на следующий го… | 01_otpuska.md | 01_otpuska.md, 01_otpuska.md, 01_otpuska.md | ✅ |
| 4 | Простые | Сколько выходных дают на свадьбу сотрудника? | 01_otpuska.md | 01_otpuska.md, 05_komandirovki.md, 01_otpuska.md | ✅ |
| 5 | Простые | До какого времени нужно сообщить руководителю, что я заболел? | 02_bolnichnye.md | 02_bolnichnye.md, 02_bolnichnye.md, 02_bolnichnye.md | ✅ |
| 6 | Простые | Доплачивает ли компания больничный до полного оклада? | 02_bolnichnye.md | 02_bolnichnye.md, 02_bolnichnye.md, 02_bolnichnye.md | ✅ |
| 7 | Простые | Сколько дней здоровья можно взять без больничного листа? | 02_bolnichnye.md | 02_bolnichnye.md, 02_bolnichnye.md, 02_bolnichnye.md | ✅ |
| 8 | Простые | Сколько дней в неделю нужно быть в офисе при гибридном формате? | 03_udalennaya_rabota.md | 03_udalennaya_rabota.md, 03_udalennaya_rabota.md, 03_udalennaya_rabota.md | ✅ |
| 9 | Простые | Сколько дней в году можно работать из-за границы? | 03_udalennaya_rabota.md | 03_udalennaya_rabota.md, 01_otpuska.md, 01_otpuska.md | ✅ |
| 10 | Простые | Какую компенсацию за интернет получают удалённые сотрудники? | 03_udalennaya_rabota.md | 03_udalennaya_rabota.md, 03_udalennaya_rabota.md, 03_udalennaya_rabota.md | ✅ |
| 11 | Простые | Какой испытательный срок у руководителей? | 04_onboarding.md | 04_onboarding.md, 06_obuchenie.md, 06_obuchenie.md | ✅ |
| 12 | Простые | Какой бонус получает бадди за адаптацию новичка? | 04_onboarding.md | 04_onboarding.md, 04_onboarding.md, 04_onboarding.md | ✅ |
| 13 | Простые | Какие суточные платят в командировке по России? | 05_komandirovki.md | 05_komandirovki.md, 05_komandirovki.md, 05_komandirovki.md | ✅ |
| 14 | Простые | Какой лимит на гостиницу в Москве в командировке? | 05_komandirovki.md | 05_komandirovki.md, 05_komandirovki.md, 05_komandirovki.md | ✅ |
| 15 | Простые | В какой срок нужно сдать авансовый отчёт после командировки? | 05_komandirovki.md | 05_komandirovki.md, 05_komandirovki.md, 05_komandirovki.md | ✅ |
| 16 | Простые | Какой годовой бюджет на обучение у обычного сотрудника? | 06_obuchenie.md | 06_obuchenie.md, 06_obuchenie.md, 06_obuchenie.md | ✅ |
| 17 | Простые | Сколько компания компенсирует за занятия английским? | 06_obuchenie.md | 06_obuchenie.md, 06_obuchenie.md, 03_udalennaya_rabota.md | ✅ |
| 18 | Простые | За сколько нужно предупредить об увольнении по собственному желанию? | 07_uvolnenie.md | 07_uvolnenie.md, 07_uvolnenie.md, 06_obuchenie.md | ✅ |
| 19 | Простые | Какая компенсация положена при увольнении по соглашению сторон? | 07_uvolnenie.md | 07_uvolnenie.md, 03_udalennaya_rabota.md, 07_uvolnenie.md | ✅ |
| 20 | Простые | Какой лимит на стоматологию в полисе ДМС? | 08_dms.md | 08_dms.md, 08_dms.md, 08_dms.md | ✅ |
| 31 | Перефразированные | Сколько раз за год можно бесплатно поговорить с мозгоправом по страхо… | 08_dms.md | 08_dms.md, 08_dms.md, 08_dms.md | ✅ |
| 32 | Перефразированные | Мой последний день в компании — пятница. Во сколько у меня отключат р… | 07_uvolnenie.md | 03_udalennaya_rabota.md, 01_otpuska.md, 07_uvolnenie.md | ✅ |
| 33 | Два документа | Я заболел, находясь в командировке. До какого времени надо предупреди… | 05_komandirovki.md, 02_bolnichnye.md | 05_komandirovki.md, 02_bolnichnye.md, 05_komandirovki.md | ✅ |
| 34 | Два документа | Я работаю полностью удалённо и еду на квартальную офлайн-встречу кома… | 03_udalennaya_rabota.md, 05_komandirovki.md | 05_komandirovki.md, 05_komandirovki.md, 05_komandirovki.md | ✅ |
| 35 | Два документа | Я вышел на работу месяц назад. Могу ли я уже пользоваться ДМС и бюдже… | 08_dms.md, 06_obuchenie.md | 06_obuchenie.md, 08_dms.md, 08_dms.md | ✅ |
| 36 | Два документа | Я увольняюсь. Что будет с неиспользованным отпуском и до какого дня б… | 07_uvolnenie.md, 08_dms.md | 08_dms.md, 01_otpuska.md, 01_otpuska.md | ✅ |
| 37 | Подвох в цифрах | Правда, что из-за границы можно работать до 90 дней в году? | 03_udalennaya_rabota.md | 02_bolnichnye.md, 03_udalennaya_rabota.md, 01_otpuska.md | ✅ |
| 38 | Подвох в цифрах | Я работаю в DemoTech 6 лет. Сколько всего дней оплачиваемого отпуска … | 01_otpuska.md | 01_otpuska.md, 01_otpuska.md, 01_otpuska.md | ✅ |
| 39 | Подвох в цифрах | Сколько суточных я получу за трёхдневную командировку в Германию? | 05_komandirovki.md | 05_komandirovki.md, 05_komandirovki.md, 05_komandirovki.md | ✅ |
| 40 | Подвох в цифрах | Курсы английского стоят 14 000 рублей в месяц. Сколько мне вернёт ком… | 06_obuchenie.md | 06_obuchenie.md, 06_obuchenie.md, 03_udalennaya_rabota.md | ✅ |
