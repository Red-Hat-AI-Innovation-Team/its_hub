from manimlib import *
from common import *

class BestOfNScene(Scene):
    def construct(self):
        title = Text("Best-of-N", font_size=30)
        title.set_color(TEXT_COLOR)
        title.to_edge(UP, buff=0.25)

        budget = Text("N = 4", font_size=16)
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
        for i in range(4):
            r = labeled_box(f"Response {i+1}", color=BORDER_COLOR, width=2.0, height=0.5, font_size=16)
            responses.add(r)
        responses.arrange(DOWN, buff=0.2)
        responses.move_to(LEFT * 1.2 + DOWN * 0.3)

        arrows_in = VGroup(*[
            thin_arrow(llm.get_right(), r.get_left()) for r in responses
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in arrows_in], lag_ratio=0.08),
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in responses], lag_ratio=0.08),
            run_time=1.2,
        )
        self.wait(0.8)

        # ORM scoring
        orm_box = labeled_box("ORM", color=ACCENT_ORANGE, width=1.4, height=0.55, font_size=18)
        orm_box.move_to(RIGHT * 1.8 + UP * 2.0)
        self.play(FadeIn(orm_box, shift=DOWN * 0.3), run_time=0.8)

        scores = [0.3, 0.8, 0.5, 0.6]
        score_labels = VGroup()
        score_bars = VGroup()

        for i, (resp, score) in enumerate(zip(responses, scores)):
            s_fill = ACCENT_GREEN if score == max(scores) else TEXT_COLOR
            s_label = Text(f"{score}", font_size=16)
            s_label.set_color(s_fill)
            s_label.next_to(resp, RIGHT, buff=0.6)
            score_labels.add(s_label)

            bar_width = score * 1.8
            bar = Rectangle(
                width=bar_width, height=0.13,
                stroke_width=0,
                fill_color=ACCENT_GREEN if score == max(scores) else INACTIVE_GRAY,
                fill_opacity=0.7,
            )
            bar.next_to(s_label, RIGHT, buff=0.15)
            bar.align_to(s_label, LEFT)
            bar.shift(RIGHT * 0.4)
            score_bars.add(bar)

        orm_arrows = VGroup(*[
            thin_arrow(orm_box.get_bottom(), resp.get_right() + UP * 0.05, color=ACCENT_ORANGE)
            for resp in responses
        ])
        self.play(
            LaggedStart(*[ShowCreation(a) for a in orm_arrows], lag_ratio=0.1),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.1),
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in score_bars], lag_ratio=0.1),
            run_time=1.5,
        )
        self.wait(1.5)

        # Highlight best, fade rest
        best_idx = scores.index(max(scores))
        fade_anims = []
        for i, resp in enumerate(responses):
            if i == best_idx:
                fade_anims.append(resp[0].animate.set_stroke(ACCENT_GREEN, width=3))
            else:
                fade_anims.append(resp.animate.set_opacity(0.3))
                fade_anims.append(score_labels[i].animate.set_opacity(0.3))
                fade_anims.append(score_bars[i].animate.set_opacity(0.3))

        self.play(*fade_anims, run_time=1.5)

        # Winner on the right
        winner = labeled_box("✓ Best Response", color=ACCENT_GREEN, width=2.4, height=0.55, font_size=18)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(RIGHT * 5.5 + responses[best_idx].get_center()[1] * UP)

        winner_arrow = thin_arrow(score_bars[best_idx].get_right() + RIGHT * 0.15, winner.get_left(), color=ACCENT_GREEN)
        self.play(ShowCreation(winner_arrow), FadeIn(winner, shift=LEFT * 0.2), run_time=1.0)
        self.wait(4.0)
