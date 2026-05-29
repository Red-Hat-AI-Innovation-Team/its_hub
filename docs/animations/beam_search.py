from manimlib import *
from common import *

class BeamSearchScene(Scene):
    def construct(self):
        title = Text("Beam Search", font_size=30)
        title.set_color(TEXT_COLOR)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.8)

        step_x = [-4.0, -0.5, 3.0]

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.6, height=0.55)
        prompt.move_to(LEFT * 6.2)
        self.play(FadeIn(prompt), run_time=0.8)

        prm_label = labeled_box("PRM", color=ACCENT_ORANGE, width=1.2, height=0.5, font_size=18)
        prm_label.move_to(UP * 3.2)
        self.play(FadeIn(prm_label), run_time=0.6)
        self.wait(0.5)

        prev_nodes = [prompt]
        children_per_level = [4, 2, 2]
        step_labels_text = ["Step 1", "Step 2", "Step 3"]

        for level in range(3):
            x = step_x[level]
            cpp = children_per_level[level]
            candidates = []
            all_arrows = VGroup()
            for parent in prev_nodes:
                for j in range(cpp):
                    idx = len(candidates)
                    node = labeled_box(f"C{idx+1}", color=BORDER_COLOR, width=0.9, height=0.4, font_size=14)
                    candidates.append(node)

            n = len(candidates)
            spacing = min(1.6, 5.5 / max(n, 1))
            for i, node in enumerate(candidates):
                y = (i - (n - 1) / 2) * spacing
                node.move_to(RIGHT * x + UP * y)

            for pi, parent in enumerate(prev_nodes):
                for ci in range(cpp):
                    child = candidates[pi * cpp + ci]
                    arr = thin_arrow(parent.get_right(), child.get_left())
                    all_arrows.add(arr)

            self.play(
                LaggedStart(*[ShowCreation(a) for a in all_arrows], lag_ratio=0.05),
                LaggedStart(*[FadeIn(c) for c in candidates], lag_ratio=0.05),
                run_time=1.0,
            )

            # PRM arrows to candidates (temporary, like Best-of-N style)
            prm_arrows = VGroup(*[
                thin_arrow(prm_label.get_bottom(), c.get_top(), color=ACCENT_ORANGE)
                for c in candidates
            ])
            self.play(
                LaggedStart(*[ShowCreation(a) for a in prm_arrows], lag_ratio=0.08),
                run_time=0.8,
            )

            scores = []
            score_labels = VGroup()
            for i, node in enumerate(candidates):
                s = [0.7, 0.3, 0.8, 0.4][i % 4] if level < 2 else [0.9, 0.5, 0.6, 0.2][i % 4]
                scores.append(s)
                best = s == max([0.7, 0.3, 0.8, 0.4][j % 4] if level < 2 else [0.9, 0.5, 0.6, 0.2][j % 4] for j in range(n))
                sl = Text(f"{s}", font_size=14)
                sl.set_color(ACCENT_GREEN if best else TEXT_COLOR)
                sl.next_to(node, RIGHT, buff=0.12)
                score_labels.add(sl)

            self.play(
                LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.05),
                run_time=0.8,
            )
            self.wait(0.8)

            # Fade PRM arrows
            self.play(*[FadeOut(a) for a in prm_arrows], run_time=0.3)

            # Prune: keep top beam_width=2
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
                    fade_anims.append(candidates[i][0].animate.set_stroke(ACCENT_GREEN, width=2.5))

            if fade_anims:
                self.play(*fade_anims, run_time=1.0)

            if level < 2:
                dup_text = Text("duplicate", font_size=14)
                dup_text.set_color(ACCENT_ORANGE)
                dup_text.move_to(RIGHT * x + DOWN * 3.0)
                self.play(FadeIn(dup_text), run_time=0.4)
                self.play(FadeOut(dup_text), run_time=0.4)

            prev_nodes = [candidates[i] for i in ranked[:beam_width]]

        self.wait(0.5)
        winner = labeled_box("✓ Best Path", color=ACCENT_GREEN, width=2.0, height=0.5, font_size=18)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(RIGHT * 5.5)

        self.play(FadeIn(winner, shift=LEFT * 0.2), run_time=1.0)
        self.wait(4.0)
