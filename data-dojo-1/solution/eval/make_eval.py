import json
import pickle

DATA = [
    {
        "src": (
            "If you want to contribute to a GitHub project, you usually start by "
            "forking the repository. The \"Fork\" button in the top-right corner "
            "creates your own copy, where you can experiment freely. Once your "
            "changes are ready, you open a pull request so the maintainers can "
            "review them. They may ask for edits before they merge your work."
        ),
        "reference": (
            "Если вы хотите внести вклад в проект на GitHub, обычно вы начинаете с "
            "форка репозитория. Кнопка «Fork» в правом верхнем углу создаёт вашу "
            "собственную копию, в которой можно свободно экспериментировать. Когда "
            "изменения готовы, вы открываете пул-реквест, чтобы мейнтейнеры могли "
            "их проверить. Они могут попросить внести правки, прежде чем примут "
            "вашу работу."
        ),
    },
    {
        "src": (
            "In the interview, the minister said that she had warned about the risks "
            "long before the crisis began. She insisted that her team had done "
            "everything possible, but that the warnings were ignored. When a "
            "journalist pressed her, she refused to name those responsible."
        ),
        "reference": (
            "В интервью министр заявила, что предупреждала о рисках задолго до начала "
            "кризиса. Она настаивала, что её команда сделала всё возможное, но "
            "предупреждения проигнорировали. Когда журналист стал настаивать, она "
            "отказалась называть виновных."
        ),
    },
    {
        "src": (
            "The new telescope can observe galaxies that formed shortly after the Big "
            "Bang. Because their light has travelled for billions of years, "
            "astronomers effectively see them as they were in the distant past. Each "
            "new image forces scientists to revise their models of how the universe "
            "evolved."
        ),
        "reference": (
            "Новый телескоп способен наблюдать галактики, образовавшиеся вскоре после "
            "Большого взрыва. Поскольку их свет шёл к нам миллиарды лет, астрономы по "
            "сути видят их такими, какими они были в далёком прошлом. Каждое новое "
            "изображение заставляет учёных пересматривать модели того, как "
            "развивалась Вселенная."
        ),
    },
    {
        "src": (
            "Anna had never left her village before. When she arrived in the city, she "
            "felt both excited and afraid. The streets were louder than anything she "
            "had imagined, and for a moment she wanted to turn back. But she "
            "remembered why she had come, and she kept walking."
        ),
        "reference": (
            "Анна никогда раньше не покидала свою деревню. Когда она приехала в город, "
            "она почувствовала одновременно и волнение, и страх. Улицы были шумнее "
            "всего, что она могла себе представить, и на мгновение ей захотелось "
            "повернуть назад. Но она вспомнила, зачем приехала, и продолжила идти."
        ),
    },
    {
        "src": (
            "To install the application, download the setup file and double-click it. "
            "In the wizard, click \"Next\", accept the license agreement, and choose "
            "the installation folder. When the progress bar reaches the end, press "
            "\"Finish\". The program will then appear in your Start menu."
        ),
        "reference": (
            "Чтобы установить приложение, скачайте установочный файл и дважды "
            "щёлкните по нему. В мастере установки нажмите «Далее», примите "
            "лицензионное соглашение и выберите папку установки. Когда индикатор "
            "выполнения дойдёт до конца, нажмите «Готово». После этого программа "
            "появится в меню «Пуск»."
        ),
    },
    {
        "src": (
            "The company announced that it would cut prices across its entire product "
            "line. Analysts were surprised, because the firm had raised prices only a "
            "year earlier. Its chief executive explained that the decision was meant "
            "to win back customers who had switched to cheaper rivals."
        ),
        "reference": (
            "Компания объявила, что снизит цены на всю свою линейку продуктов. "
            "Аналитики были удивлены, ведь всего годом ранее фирма цены повышала. Её "
            "генеральный директор объяснил, что это решение призвано вернуть "
            "клиентов, перешедших к более дешёвым конкурентам."
        ),
    },
    {
        "src": (
            "Machine translation models handle single sentences remarkably well. "
            "However, when they translate whole paragraphs, they often lose track of "
            "context: pronouns no longer agree, the tone shifts abruptly, and the "
            "same term is rendered differently in different places. Solving this "
            "requires the model to treat the text as a coherent whole."
        ),
        "reference": (
            "Модели машинного перевода прекрасно справляются с отдельными "
            "предложениями. Однако при переводе целых абзацев они часто теряют "
            "контекст: местоимения перестают согласовываться, тон резко меняется, а "
            "один и тот же термин в разных местах переводится по-разному. Чтобы "
            "решить эту проблему, модель должна воспринимать текст как единое связное "
            "целое."
        ),
    },
    {
        "src": (
            "He told reporters that he was, in his own words, \"made out of metal\". "
            "Everyone laughed, assuming it was a joke about his calm under pressure. "
            "Only later did they realise he had been talking about his new prosthetic "
            "arm, which he wore with obvious pride."
        ),
        "reference": (
            "Он сказал журналистам, что он, по его собственным словам, «сделан из "
            "металла». Все засмеялись, решив, что это шутка о его невозмутимости под "
            "давлением. Лишь позже они поняли, что он говорил о своём новом протезе "
            "руки, который носил с явной гордостью."
        ),
    },
]


def main() -> None:
    rows = [{"rid": i, "src": d["src"]} for i, d in enumerate(DATA)]
    refs = [{"rid": i, "reference": d["reference"]} for i, d in enumerate(DATA)]
    with open("eval-input.pickle", "wb") as f:
        pickle.dump(rows, f)
    with open("references.json", "w") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(rows)} examples -> eval-input.pickle, references.json")


if __name__ == "__main__":
    main()
