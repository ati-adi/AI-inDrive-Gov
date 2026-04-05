import os
import json
import re
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.3-70b-versatile"

K2_API_URL = "https://api.k2think.ai/v1/chat/completions"
K2_API_KEY = os.environ.get("K2_API_KEY", "IFM-YImFDC2XhobEfvNE")
K2_MODEL = "MBZUAI-IFM/K2-Think-v2"

USE_GROQ = True


def strip_thinking(text):
    if not text:
        return "Не удалось сгенерировать объяснение."
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text).strip()
    markers = ["Thus answer:", "Thus final answer:", "Final answer:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            return text.strip()
    if len(text) > 500:
        paragraphs = text.split("\n\n")
        russian_blocks = []
        for p in reversed(paragraphs):
            p = p.strip()
            if not p:
                continue
            cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', p))
            total = max(len(p), 1)
            if cyrillic / total > 0.3 and len(p) > 50:
                russian_blocks.insert(0, p)
            elif russian_blocks:
                break
        if russian_blocks:
            return "\n\n".join(russian_blocks).strip()
    return text.strip()


SYSTEM_PROMPT = """Ты — AI-советник для государственной системы субсидирования животноводства в Казахстане (ГИСС).

Нормативная база: Приказ МСХ РК №108 от 15.03.2019 (ред. №428 от 18.11.2025).

Ключевые нормативы:
- Субсидия на племенных животных <= 50% стоимости приобретения
- Обязательство сохранности: маточное 2 года, производители 18 месяцев
- Срок подачи: 20 января - 20 декабря
- Нормы падежа: КРС мясное 2%, молочное 3%, овцы 3%, лошади 2.3%

Формат ответа (строго):
1. Одно предложение - рекомендация (одобрить / отклонить)
2. 2-3 причины почему такой балл - со ссылками на конкретные цифры
3. Красные флаги (если есть)
4. Статус фермера (новый/опытный, стадо растет/падает)

Ограничения:
- Кратко, 4-6 предложений
- Без технического жаргона (не XGBoost, не Isolation Forest)
- Говори "прогноз по истории", "показатель надежности", "показатель региона"
- Суммы в тенге, количество в головах
- НЕ принимай решение, только объясни факты"""


CHAT_SYSTEM = """Ты — AI-советник для государственной системы субсидирования животноводства в Казахстане.

Тебе дана информация о конкретной заявке на субсидию. Чиновник задает вопросы:
- Почему балл низкий/высокий?
- Что не так с фермером?
- Какие нарушения?
- Можно ли одобрить несмотря на низкий балл?

Отвечай кратко (2-4 предложения), по делу, со ссылками на цифры.
Не используй технический жаргон.
Говори "прогноз по истории", "показатель надежности", "показатель региона".
Суммы в тенге, количество в головах.
Ты советник, не принимай решение за чиновника."""


def build_prompt(score_card, rule_violations=None, isj_data=None):
    parts = []
    parts.append(f"Заявка на субсидию:")
    parts.append(f"- Район: {score_card.get('district', 'неизвестно')}")
    parts.append(f"- Направление: {score_card.get('direction', 'неизвестно')}")
    parts.append(f"- Сумма: {score_card.get('amount', 0):,.0f} тенге")
    parts.append(f"- Количество: {score_card.get('unit_count', 0):.0f} голов/единиц")
    parts.append(f"")
    parts.append(f"Итоговый балл: {score_card.get('final_score', 0):.0f} из 100")
    parts.append(f"Рекомендация: {score_card.get('recommendation', 'unknown')}")
    parts.append(f"")
    parts.append("Компоненты оценки:")
    for comp in score_card.get("components", []):
        parts.append(f"  - {comp['name']}: {comp['score']:.0f}/100 (вес {comp['weight']:.0%})")
    if rule_violations:
        parts.append(f"\nНарушения правил: {', '.join(rule_violations)}")
    if isj_data:
        parts.append(f"\nДанные о фермере:")
        if isj_data.get("is_new_farmer"):
            parts.append(f"  - Новый фермер (первая заявка)")
        else:
            parts.append(f"  - Активен {isj_data.get('years_active', '?')} лет")
            parts.append(f"  - Стадо: {isj_data.get('herd_trend', '?')}")
            parts.append(f"  - Надежность: {isj_data.get('reliability_score', 0):.0%}")
            if isj_data.get("herd_declining"):
                parts.append(f"  - СТАДО СОКРАЩАЕТСЯ")
        if isj_data.get("expected_mortality") and isj_data.get("actual_mortality"):
            exp = isj_data["expected_mortality"]
            act = isj_data["actual_mortality"]
            parts.append(f"  - Норма падежа: {exp:.1%}, фактический: {act:.1%}")
            if act > exp * 1.5:
                parts.append(f"  - Падеж ВЫШЕ нормы в {act/max(exp,0.001):.1f} раз")
        if isj_data.get("pasture_norm_ha"):
            parts.append(f"  - Норма пастбищ: {isj_data['pasture_norm_ha']} га/голова")
    parts.append("\nОбъясни чиновнику кратко, почему такой балл.")
    return "\n".join(parts)


def _call_llm(messages):
    if USE_GROQ:
        url, key, model = GROQ_API_URL, GROQ_API_KEY, GROQ_MODEL
    else:
        url, key, model = K2_API_URL, K2_API_KEY, K2_MODEL

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        return strip_thinking(raw)
    except requests.exceptions.RequestException as e:
        return f"[Ошибка API: {str(e)}]"
    except (KeyError, IndexError) as e:
        return f"[Ошибка: {str(e)}]"


def get_explanation(score_card, rule_violations=None, isj_data=None):
    prompt = build_prompt(score_card, rule_violations, isj_data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return _call_llm(messages)


def chat_about_application(score_card, history, user_message, rule_violations=None, isj_data=None):
    context = build_prompt(score_card, rule_violations, isj_data)
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": f"Данные заявки:\n\n{context}"},
        {"role": "assistant", "content": "Понял, готов ответить на вопросы по этой заявке."},
    ]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return _call_llm(messages)
