from manimlib import *
from common import *

class BeamSearchScene(Scene):
    def construct(self):
        title = Text("Beam Search", font_size=30)
        title.set_fill(TEXT_COLOR)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.4)

        level_y = [2.0, 0.0, -2.0]

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.6, height=0.55)
        prompt.move_to(UP * 3.0)
        self.play(FadeIn(prompt), run_time=0.4)

        prm_label = labeled_box("PRM", color=ACCENT_ORANGE, width=1.2, height=0.5, font_size=18)
        prm_label.move_to(RIGHT * 5.5 + UP * 3.0)
        self.play(FadeIn(prm_label), run_time=0.3)

        prev_nodes = [prompt]
        children_per_level = [4, 2, 2]

        for level in range(3):
            y = level_y[level]
            cpp = children_per_level[level]
            candidates = []
            all_arrows = VGroup()
            for parent in prev_nodes:
                for j in range(cpp):
                    node = Dot(radius=0.15, fill_color=ACCENT_BLUE, fill_opacity=0.8)
                    candidates.append(node)

            n = len(candidates)
            spacing = min(2.5, 10.0 / max(n, 1))
            for i, node in enumerate(candidates):
                x = (i - (n - 1) / 2) * spacing
                node.move_to(RIGHT * x + UP * y)

            for pi, parent in enumerate(prev_nodes):
                for ci in range(cpp):
                    child = candidates[pi * cpp + ci]
                    arr = thin_arrow(parent.get_bottom(), child.get_top())
                    all_arrows.add(arr)

            self.play(
                LaggedStart(*[ShowCreation(a) for a in all_arrows], lag_ratio=0.05),
                LaggedStart(*[FadeIn(c) for c in candidates], lag_ratio=0.05),
                run_time=0.6,
            )

            scores = []
            score_labels = VGroup()
            for i, node in enumerate(candidates):
                s = [0.7, 0.3, 0.8, 0.4][i % 4] if level < 2 else [0.9, 0.5, 0.6, 0.2][i % 4]
                scores.append(s)
                sl = Text(f"{s}", font_size=14)
                sl.set_fill(TEXT_COLOR)
                sl.next_to(node, DOWN, buff=0.1)
                score_labels.add(sl)

            self.play(
                LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.05),
                run_time=0.4,
            )

            beam_width = 2
            ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
            keep = set(ranked[:beam_width])

            fade_anims = []
            for i in range(n):
                if i not in keep:
                    fade_anims.append(candidates[i].animate.set_opacity(0.15))
                    fade_anims.append(score_labels[i].animate.set_opacity(0.15))
                    arr_idx = i
                    if arr_idx < len(all_arrows):
                        fade_anims.append(all_arrows[arr_idx].animate.set_opacity(0.15))
                else:
                    fade_anims.append(candidates[i].animate.set_fill(ACCENT_GREEN))

            if fade_anims:
                self.play(*fade_anims, run_time=0.5)

            if level < 2:
                dup_text = Text("duplicate", font_size=14)
                dup_text.set_fill(ACCENT_ORANGE)
                dup_text.move_to(RIGHT * 5.5 + UP * y)
                self.play(FadeIn(dup_text), run_time=0.2)
                self.play(FadeOut(dup_text), run_time=0.2)

            prev_nodes = [candidates[i] for i in ranked[:beam_width]]

        winner = labeled_box("✓ Best Path", color=ACCENT_GREEN, width=2.0, height=0.5, font_size=18)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(DOWN * 3.2)

        self.play(FadeIn(winner, shift=UP * 0.2), run_time=0.5)
        self.wait(1.5)
