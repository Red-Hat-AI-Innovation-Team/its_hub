from manimlib import *
from common import *

class SelfConsistencyScene(Scene):
    def construct(self):
        title = Text("Self-Consistency", font_size=30)
        title.set_color(TEXT_COLOR)
        title.to_edge(UP, buff=0.25)

        budget = Text("N = 5", font_size=16)
        budget.set_color(INACTIVE_GRAY)
        budget.next_to(title, DOWN, buff=0.15)

        self.play(FadeIn(title), FadeIn(budget), run_time=0.8)

        # Prompt → LLM → Responses
        prompt = labeled_box("Prompt", color=PROMPT_COLOR, width=1.6, height=0.55, fill=PROMPT_FILL)
        prompt.move_to(LEFT * 6 + DOWN * 0.3)
        self.play(FadeIn(prompt), run_time=0.6)

        llm = labeled_box("LLM", color=LLM_COLOR, width=1.2, height=0.55, font_size=18, fill=LLM_FILL)
        llm.move_to(LEFT * 4 + DOWN * 0.3)
        prompt_to_llm = thin_arrow(prompt.get_right(), llm.get_left())
        self.play(ShowCreation(prompt_to_llm), FadeIn(llm), run_time=0.6)

        responses = VGroup()
        for i in range(5):
            r = labeled_box(f"Response {i+1}", color=BORDER_COLOR, width=2.0, height=0.45, font_size=14)
            responses.add(r)
        responses.arrange(DOWN, buff=0.12)
        responses.move_to(LEFT * 1 + DOWN * 0.3)

        arrows_in = VGroup(*[
            thin_arrow(llm.get_right(), r.get_left()) for r in responses
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in arrows_in], lag_ratio=0.06),
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in responses], lag_ratio=0.06),
            run_time=1.2,
        )
        self.wait(0.5)

        # Show extracted answers inline
        answers = ["7", "12", "7", "7", "12"]
        answer_labels = VGroup()
        for i, (resp, ans) in enumerate(zip(responses, answers)):
            fill = ACCENT_GREEN if ans == "7" else ACCENT_BLUE
            lbl = Text(f"→ {ans}", font_size=18)
            lbl.set_color(fill)
            lbl.next_to(resp, RIGHT, buff=0.3)
            answer_labels.add(lbl)

        self.play(
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in answer_labels], lag_ratio=0.1),
            run_time=1.5,
        )
        self.wait(1.0)

        # Highlight matching answers
        self.play(
            *[resp[0].animate.set_stroke(
                ACCENT_GREEN if answers[i] == "7" else ACCENT_BLUE, width=2.5
            ) for i, resp in enumerate(responses)],
            run_time=1.2,
        )
        self.wait(1.0)

        # Vote tally
        vote_7 = Text("7  →  3 votes", font_size=20)
        vote_7.set_color(ACCENT_GREEN)
        vote_12 = Text("12 →  2 votes", font_size=20)
        vote_12.set_color(ACCENT_BLUE)
        tally = VGroup(vote_7, vote_12).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        tally.move_to(RIGHT * 4.5 + UP * 0.5)

        self.play(FadeIn(tally), run_time=1.2)
        self.wait(1.5)

        # Winner
        winner = labeled_box("✓ Answer: 7", color=ACCENT_GREEN, width=2.2, height=0.6, font_size=18)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(RIGHT * 4.5 + DOWN * 1.0)

        self.play(FadeIn(winner, shift=UP * 0.2), run_time=1.0)
        self.wait(4.0)
