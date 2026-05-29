from manimlib import *
from common import *
import numpy as np

PARTICLE_COLORS = ["#4A90D9", "#E06C75", "#61AFEF", "#C678DD", "#E5C07B"]

class ParticleFilteringScene(Scene):
    def construct(self):
        title = Text("Particle Filtering", font_size=30)
        title.set_color(TEXT_COLOR)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), run_time=0.8)

        prompt = labeled_box("Prompt", color=ACCENT_BLUE, width=1.6, height=0.55)
        prompt.move_to(LEFT * 6 + DOWN * 0.3)
        self.play(FadeIn(prompt), run_time=0.8)

        prm_label = labeled_box("PRM", color=ACCENT_ORANGE, width=1.2, height=0.5, font_size=18)
        prm_label.move_to(UP * 3.2)
        self.play(FadeIn(prm_label), run_time=0.6)

        step_x = [-3.5, -0.5, 2.5]
        for i, x in enumerate(step_x):
            sl = Text(f"Step {i+1}", font_size=14)
            sl.set_color(INACTIVE_GRAY)
            sl.move_to(RIGHT * x + UP * 2.5)
            self.add(sl)

        self.wait(0.5)

        n_particles = 5
        particle_y = np.linspace(1.5, -2.5, n_particles)

        # Initial particles from prompt
        particles = VGroup()
        for idx, y in enumerate(particle_y):
            p = labeled_box(f"P{idx+1}", color=PARTICLE_COLORS[idx], width=1.0, height=0.4, font_size=14)
            p.move_to(LEFT * 4.8 + UP * y)
            particles.add(p)

        init_arrows = VGroup(*[
            thin_arrow(prompt.get_right(), p.get_left()) for p in particles
        ])

        self.play(
            LaggedStart(*[ShowCreation(a) for a in init_arrows], lag_ratio=0.05),
            LaggedStart(*[FadeIn(p) for p in particles], lag_ratio=0.05),
            run_time=1.0,
        )
        self.wait(0.5)

        weights_per_step = [
            [0.3, 0.7, 0.5, 0.9, 0.2],
            [0.4, 0.8, 0.3, 0.6, 0.8],
            [0.2, 0.9, 0.5, 0.7, 0.6],
        ]
        resample_from = [
            [1, 3, 2, 3, 1],
            [1, 1, 4, 3, 4],
            None,
        ]

        current_particles = list(particles)
        current_colors = list(PARTICLE_COLORS)

        for step in range(3):
            x_target = step_x[step]
            weights = weights_per_step[step]

            # Generate new particles at this step
            new_particles = VGroup()
            move_arrows = VGroup()
            for i, p in enumerate(current_particles):
                new_p = labeled_box(f"P{i+1}", color=current_colors[i], width=1.0, height=0.4, font_size=14)
                new_p.move_to(RIGHT * x_target + UP * particle_y[i])
                new_particles.add(new_p)
                arr = thin_arrow(p.get_right(), new_p.get_left())
                move_arrows.add(arr)

            self.play(
                LaggedStart(*[ShowCreation(a) for a in move_arrows], lag_ratio=0.04),
                LaggedStart(*[FadeIn(p) for p in new_particles], lag_ratio=0.04),
                run_time=0.8,
            )

            # PRM scoring arrows + score labels (Best-of-N style)
            prm_arrows = VGroup(*[
                thin_arrow(prm_label.get_bottom(), p.get_top(), color=ACCENT_ORANGE)
                for p in new_particles
            ])
            self.play(
                LaggedStart(*[ShowCreation(a) for a in prm_arrows], lag_ratio=0.08),
                run_time=0.8,
            )

            score_labels = VGroup()
            for i, (p, w) in enumerate(zip(new_particles, weights)):
                best = w == max(weights)
                s_color = ACCENT_GREEN if best else TEXT_COLOR
                sl = Text(f"{w}", font_size=14)
                sl.set_color(s_color)
                sl.next_to(p, RIGHT, buff=0.15)
                score_labels.add(sl)

            self.play(
                LaggedStart(*[FadeIn(s) for s in score_labels], lag_ratio=0.08),
                run_time=0.8,
            )
            self.wait(0.8)

            # Fade PRM arrows after scoring
            self.play(*[FadeOut(a) for a in prm_arrows], run_time=0.3)

            # Resample (not on last step)
            if resample_from[step] is not None:
                sources = resample_from[step]
                surviving_sources = set(sources)
                eliminated = [i for i in range(n_particles) if i not in surviving_sources]

                # Fade eliminated particles and their scores
                fade_anims = []
                for i in eliminated:
                    fade_anims.append(new_particles[i].animate.set_opacity(0.15))
                    fade_anims.append(score_labels[i].animate.set_opacity(0.15))

                if fade_anims:
                    self.play(*fade_anims, run_time=0.6)

                # Duplicate survivors into vacated slots
                dup_anims = []
                resampled = list(new_particles)
                new_colors = list(current_colors)
                for i in range(n_particles):
                    src = sources[i]
                    if src != i:
                        copy_p = labeled_box(f"P{src+1}", color=current_colors[src], width=1.0, height=0.4, font_size=14)
                        copy_p.move_to(new_particles[src].get_center())
                        self.add(copy_p)
                        dup_anims.append(
                            copy_p.animate.move_to(new_particles[i].get_center())
                        )
                        resampled[i] = copy_p
                        new_colors[i] = current_colors[src]

                if dup_anims:
                    self.play(*dup_anims, run_time=0.8)

                current_particles = resampled
                current_colors = new_colors
            else:
                current_particles = list(new_particles)

            self.wait(0.3)

        # Select best particle
        self.wait(0.5)
        best_idx = weights_per_step[2].index(max(weights_per_step[2]))
        best_particle = current_particles[best_idx]

        winner = labeled_box("✓ Best", color=ACCENT_GREEN, width=1.4, height=0.45, font_size=16)
        winner[0].set_fill(ACCENT_GREEN, opacity=0.15)
        winner.move_to(RIGHT * 5 + UP * particle_y[best_idx])

        winner_arrow = thin_arrow(best_particle.get_right(), winner.get_left(), color=ACCENT_GREEN)
        self.play(ShowCreation(winner_arrow), FadeIn(winner), run_time=1.0)
        self.wait(4.0)
