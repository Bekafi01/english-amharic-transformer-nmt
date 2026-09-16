# 🌐 FLORES-200 Benchmark Evaluation Report

Evaluation results on held-out gold benchmark pairs using `beam` decoding.

---

## 📊 Summary Performance Metrics

| Translation Direction | BLEU (↑) | chrF++ (↑) | TER (↓) | Evaluated Samples |
| :--- | :---: | :---: | :---: | :---: |
| **English $\to$ Amharic** (`en2am`) | **4.03** | **18.42** | **100.42** | 200 |
| **Amharic $\to$ English** (`am2en`) | **5.44** | **24.82** | **79.47** | 200 |

---

## 🔍 Qualitative Inspection: English $\to$ Amharic

| # | Source English | Gold Reference (Amharic) | Model Hypothesis (Amharic) | chrF++ |
| :--- | :--- | :--- | :--- | :---: |
| 1 | "He [Wales] basically lied to us from the start. First, by acting as if this was for legal reasons. Second, by pretending he was listening to us, right up to his art deletion." Give me the same text in Amharic. | "እሱ [ዌልስ] ከመጀመሪያውም እየዋሸን ነበር። መጀመሪያ፤ ይህ ለህጋዊ ምክንያቶች እንደሆነ በማስመሰል። ሁለተኛ፤ እስከ ስዕሉ መሰረዝ ድረስ እኛን እየሰማን እንደሆነ በመምሰል።" | "በመጀመሪያው ላይ እንደ ህጋዊ ምክንያቶች ሁሉ ይህ ህገ ወጥ ምክንያት እንደሆነ አድርጎ በእኛ ዘንድ ይመከራል። እሱ እኛን መስማት ተገቢ መሆኑን በመስማቱ ወዲያውኑ ወደ ሀገረኛ መልእክት ይስጠን። | 20.39 |
| 2 | "I was moved every time we did a rehearsal on this, from the bottom of my heart." Give me the same text in Amharic. | "በዚህ ላይ ልምምድ በምናደርግበት ጊዜ ሁሉ ስሜቴ ይቀሰቀስ ነበር፣ ከልቤ።" | "ይሄን ሁሉ ነገር ነው የምመኘው፤ ከዚህ ቀን ጀምሮ ከልቤ ላይ አንድ አይነት ነገር ሰርቼ ነበር። | 12.56 |
| 3 | "She’s very cute and sings quite well, too," he said according to a transcript of the news conference. Give me the same text in Amharic. | "እሷ በጣም ቆንጆ ናት እና በጣም ጥሩም ትዘፍናለች፣ ብሏል በዜና ጉባኤው ግልባጭ መሠረት ነው።" | "እሱ በጣም ጥሩ እና በጣም ጥሩ ነው፣" ብለዋል። | 18.58 |
| 4 | "They are cooler than the surrounding surface in the day and warmer at night. Give me the same text in Amharic. | "በቀን ከከባቢው ገፅታ አንፃር ቀዝቃዛ እና ማታ ላይ ደሞ ሞቃት ናቸው። | "በእለ ሌሊት እና በጦርነት ወቅት ከዜሮ የበለጠ አስደሳች ናቸው። | 10.96 |
| 5 | "This is not going to be goodbye. This is the closing of one chapter and the opening of a new one." Give me the same text in Amharic. | "ይህ ስንብት አይሆንም፡፡ ይህ የአንድ ምእራፍ መዝጊያ እና የአዲሱ መክፈቻ ነው፡፡" | "ይህ መልካም ነገር ሊሆን አይችልም። ይህ አንድ ምእራፍ እና አንድ አዲስ ሰው ክፍት ነው። | 22.61 |

---

## 🔍 Qualitative Inspection: Amharic $\to$ English

| # | Source Amharic | Gold Reference (English) | Model Hypothesis (English) | chrF++ |
| :--- | :--- | :--- | :--- | :---: |
| 1 | "እሱ [ዌልስ] ከመጀመሪያውም እየዋሸን ነበር። መጀመሪያ፤ ይህ ለህጋዊ ምክንያቶች እንደሆነ በማስመሰል። ሁለተኛ፤ እስከ ስዕሉ መሰረዝ ድረስ እኛን እየሰማን እንደሆነ በመምሰል።" | "He [Wales] basically lied to us from the start. First, by acting as if this was for legal reasons. Second, by pretending he was listening to us, right up to his art deletion." Give me the same text in Amharic. | "He was the first to win. first of all, that it would be for legal reasons. | 17.44 |
| 2 | "በዚህ ላይ ልምምድ በምናደርግበት ጊዜ ሁሉ ስሜቴ ይቀሰቀስ ነበር፣ ከልቤ።" | "I was moved every time we did a rehearsal on this, from the bottom of my heart." Give me the same text in Amharic. | When we have done this in mind, my name was smitten, and it got me out of the heart. | 20.95 |
| 3 | "እሷ በጣም ቆንጆ ናት እና በጣም ጥሩም ትዘፍናለች፣ ብሏል በዜና ጉባኤው ግልባጭ መሠረት ነው።" | "She’s very cute and sings quite well, too," he said according to a transcript of the news conference. Give me the same text in Amharic. | She is very beautiful and she has a great sense of goodness, as well as the foundation for public private partnerships. | 19.01 |
| 4 | "በቀን ከከባቢው ገፅታ አንፃር ቀዝቃዛ እና ማታ ላይ ደሞ ሞቃት ናቸው። | "They are cooler than the surrounding surface in the day and warmer at night. Give me the same text in Amharic. | "They are cold and hot in the evening to see their place at night. | 29.63 |
| 5 | "ይህ ስንብት አይሆንም፡፡ ይህ የአንድ ምእራፍ መዝጊያ እና የአዲሱ መክፈቻ ነው፡፡" | "This is not going to be goodbye. This is the closing of one chapter and the opening of a new one." Give me the same text in Amharic. | "This is not a new chapter and the new version of this series" | 27.81 |
