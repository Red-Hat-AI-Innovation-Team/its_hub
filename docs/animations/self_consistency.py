from manimlib import *
from common import *

class SelfConsistencyScene(Scene):
    def construct(self):
        title = Text("Self-Consistency", font_size=30)
        title.set_color(TEXT_COLOR)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.8)

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.8, height=0.6)
        prompt.move_to(LEFT * 5.5 + DOWN * 0.2)
        self.play(FadeIn(prompt), run_time=0.8)

        raw_labels = ["R1", "R2", "R3", "R4", "R5"]
        responses = VGroup()
        for label in raw_labels:
            r = labeled_box(label, color=BORDER_COLOR, width=1.0, height=0.45, font_size=18)
            responses.add(r)
        responses.arrange(DOWN, buff=0.15)
        responses.move_to(LEFT * 3 + DOWN * 0.2)

        arrows_in = VGroup(*[
            thin_arrow(prompt.get_right(), r.get_left()) for r in responses
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in arrows_in], lag_ratio=0.06),
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in responses], lag_ratio=0.06),
            run_time=1.5,
        )

        proj = labeled_box("Extract\nAnswer", color=ACCENT_ORANGE, width=1.5, height=0.7, font_size=16)
        proj.move_to(LEFT * 0.8 + DOWN * 0.2)

        arrows_proj = VGroup(*[
            thin_arrow(r.get_right(), proj.get_left()) for r in responses
        ])

        self.play(
            FadeIn(proj),
            LaggedStart(*[ShowCreation(a) for a in arrows_proj], lag_ratio=0.04),
            run_time=1.2,
        )

        answers = ["7", "12", "7", "7", "12"]
        answer_labels = VGroup()
        for i, ans in enumerate(answers):
            fill = ACCENT_GREEN if ans == "7" else ACCENT_BLUE
            lbl = Text(ans, font_size=22)
            lbl.set_color(fill)
            lbl.move_to(RIGHT * 1.2 + DOWN * 0.2 + UP * (1.0 - i * 0.5))
            answer_labels.add(lbl)

        arrows_out = VGroup(*[
            thin_arrow(proj.get_right(), lbl.get_left()) for lbl in answer_labels
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in arrows_out], lag_ratio=0.04),
            LaggedStart(*[FadeIn(l) for l in answer_labels], lag_ratio=0.04),
            run_time=1.2,
        )

        vote_7 = Text("7  →  3 votes", font_size=20)
        vote_7.set_color(ACCENT_GREEN)
        vote_12 = Text("12 →  2 votes", font_size=20)
        vote_12.set_color(ACCENT_BLUE)
        tally = VGroup(vote_7, vote_12).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        tally.move_to(RIGHT * 3.8 + UP * 0.3)

        self.play(FadeIn(tally), run_time=1.2)

        winner = labeled_box("✓ Answer: 7", color=ACCENT_GREEN, width=2.2, height=0.6, font_size=20)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(RIGHT * 3.8 + DOWN * 0.8)

        self.play(FadeIn(winner, shift=UP * 0.2), run_time=1.0)
        self.wait(3.0)
