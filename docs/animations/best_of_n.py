from manimlib import *
from common import *

class BestOfNScene(Scene):
    def construct(self):
        title = Text("Best-of-N", font_size=30)
        title.set_fill(TEXT_COLOR)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.4)

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.8, height=0.6)
        prompt.move_to(LEFT * 5 + DOWN * 0.2)
        self.play(FadeIn(prompt), run_time=0.4)

        responses = VGroup()
        for i in range(4):
            r = labeled_box(f"Response {i+1}", color=BORDER_COLOR, width=2.0, height=0.5, font_size=18)
            responses.add(r)
        responses.arrange(DOWN, buff=0.2)
        responses.move_to(LEFT * 1.5 + DOWN * 0.2)

        arrows_in = VGroup(*[
            thin_arrow(prompt.get_right(), r.get_left()) for r in responses
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in arrows_in], lag_ratio=0.08),
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in responses], lag_ratio=0.08),
            run_time=0.8,
        )

        orm_box = labeled_box("ORM", color=ACCENT_ORANGE, width=1.4, height=0.6, font_size=20)
        orm_box.move_to(RIGHT * 1.5 + UP * 1.8)
        self.play(FadeIn(orm_box, shift=DOWN * 0.3), run_time=0.5)

        scores = [0.3, 0.8, 0.5, 0.6]
        score_labels = VGroup()
        score_bars = VGroup()

        for i, (resp, score) in enumerate(zip(responses, scores)):
            s_fill = ACCENT_GREEN if score == max(scores) else TEXT_COLOR
            s_label = Text(f"{score}", font_size=18)
            s_label.set_fill(s_fill)
            s_label.next_to(resp, RIGHT, buff=0.8)
            score_labels.add(s_label)

            bar_width = score * 2.0
            bar = Rectangle(
                width=bar_width, height=0.15,
                stroke_width=0,
                fill_color=ACCENT_GREEN if score == max(scores) else INACTIVE_GRAY,
                fill_opacity=0.7,
            )
            bar.next_to(s_label, RIGHT, buff=0.2)
            bar.align_to(s_label, LEFT)
            bar.shift(RIGHT * 0.5)
            score_bars.add(bar)

        orm_arrows = VGroup(*[
            thin_arrow(orm_box.get_bottom(), resp.get_right() + UP * 0.05, color=ACCENT_ORANGE)
            for resp in responses
        ])
        self.play(
            LaggedStart(*[ShowCreation(a) for a in orm_arrows], lag_ratio=0.1),
            run_time=0.6,
        )
        self.play(
            LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.1),
            LaggedStart(*[GrowFromEdge(b, LEFT) for b in score_bars], lag_ratio=0.1),
            run_time=0.8,
        )

        best_idx = scores.index(max(scores))
        fade_anims = []
        for i, resp in enumerate(responses):
            if i == best_idx:
                fade_anims.append(resp[0].animate.set_stroke(ACCENT_GREEN, width=3))
            else:
                fade_anims.append(resp.animate.set_opacity(0.3))
                fade_anims.append(score_labels[i].animate.set_opacity(0.3))
                fade_anims.append(score_bars[i].animate.set_opacity(0.3))

        self.play(*fade_anims, run_time=0.8)

        winner = labeled_box("✓ Best Response", color=ACCENT_GREEN, width=2.4, height=0.6, font_size=20)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.next_to(responses[best_idx], DOWN, buff=0.6)
        winner.shift(RIGHT * 1.5)

        self.play(FadeIn(winner, shift=UP * 0.2), run_time=0.5)
        self.wait(1.5)
