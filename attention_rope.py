from manim import *
import numpy as np

# ---------- Global B/W style ----------
config.background_color = BLACK
BW = WHITE
GREY = GREY_B

class Title(Scene):
    def construct(self):
        title = Text("Attention & RoPE", color=BW, weight=BOLD).scale(1.3)
        sub = Text("How a transformer knows what to look at", color=GREY).scale(0.5)
        sub.next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(sub, shift=UP))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(sub))


class TokensToVectors(Scene):
    def construct(self):
        words = ["The", "cat", "sat", "on", "mat"]
        boxes = VGroup(*[
            VGroup(
                Square(0.9, color=BW),
                Text(w, color=BW).scale(0.5)
            ) for w in words
        ]).arrange(RIGHT, buff=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15))
        self.wait()

        arrows = VGroup()
        vecs = VGroup()
        for b in boxes:
            arrow = Arrow(b.get_bottom(), b.get_bottom() + DOWN * 1.2, color=GREY, buff=0.1)
            vec = Matrix([["x1"], ["x2"], ["x3"]], color=BW).scale(0.4)
            vec.next_to(arrow, DOWN)
            arrows.add(arrow)
            vecs.add(vec)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(v) for v in vecs], lag_ratio=0.15))

        caption = Text("Each token becomes a vector (embedding)", color=GREY).scale(0.5)
        caption.to_edge(DOWN)
        self.play(FadeIn(caption))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class QKV(Scene):
    def construct(self):
        header = Text("Query, Key, Value", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        x = Matrix([["x"]], color=BW).scale(0.6).shift(LEFT * 5)
        self.play(FadeIn(x))

        labels = ["Q = Wq x", "K = Wk x", "V = Wv x"]
        targets = VGroup(*[
            Text(lab, color=BW).scale(0.6) for lab in labels
        ]).arrange(DOWN, buff=0.8).shift(RIGHT * 2)

        arrows = VGroup(*[
            Arrow(x.get_right(), t.get_left(), color=GREY, buff=0.2)
            for t in targets
        ])

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2))
        self.play(LaggedStart(*[Write(t) for t in targets], lag_ratio=0.2))

        note = Text(
            "Query: what am I looking for?\n"
            "Key: what do I contain?\n"
            "Value: what do I pass along?",
            color=GREY
        ).scale(0.4).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


class DotProductAttention(Scene):
    def construct(self):
        header = Text("Attention Score", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        q = Matrix([["q1"], ["q2"]], color=BW).scale(0.6).shift(LEFT * 4)
        k = Matrix([["k1"], ["k2"]], color=BW).scale(0.6).shift(LEFT * 1)
        dot = MathTex(r"\cdot", color=BW).scale(1.2).next_to(q, RIGHT, buff=0.3)
        eq = MathTex("=", color=BW).scale(1.2).next_to(k, RIGHT, buff=0.3)
        score = MathTex(r"q \cdot k", color=BW).scale(0.9).next_to(eq, RIGHT, buff=0.3)

        self.play(FadeIn(q), FadeIn(dot), FadeIn(k), FadeIn(eq), FadeIn(score))
        self.wait()

        formula = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=BW
        ).scale(0.8).shift(DOWN * 1.8)
        self.play(Write(formula))

        note = Text("Bigger dot product → tokens relate more strongly", color=GREY)
        note.scale(0.45).next_to(formula, DOWN)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


class SoftmaxBars(Scene):
    def construct(self):
        header = Text("Softmax turns scores into weights", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        values = [0.1, 0.7, 0.3, -0.2, 0.5]
        softmax_vals = np.exp(values) / np.sum(np.exp(values))

        bars = VGroup()
        labels = VGroup()
        axis_y = -1
        for i, v in enumerate(softmax_vals):
            bar = Rectangle(width=0.8, height=v * 4, color=BW, fill_color=BW, fill_opacity=0.8)
            bar.move_to(RIGHT * (i - 2) * 1.3 + UP * (axis_y + v * 2), aligned_edge=DOWN)
            lab = Text(f"{v:.2f}", color=GREY).scale(0.4).next_to(bar, UP, buff=0.1)
            bars.add(bar)
            labels.add(lab)

        baseline = Line(LEFT * 4, RIGHT * 4, color=GREY).move_to(DOWN * axis_y * -1 + DOWN * 1)
        self.play(Create(baseline))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.15))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


class PositionProblem(Scene):
    def construct(self):
        header = Text("But attention has no sense of order!", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        s1 = Text('"The cat chased the dog"', color=BW).scale(0.6)
        s2 = Text('"The dog chased the cat"', color=BW).scale(0.6)
        s1.shift(UP * 0.6)
        s2.shift(DOWN * 0.6)
        self.play(Write(s1))
        self.play(Write(s2))

        note = Text("Same words, same bag → same attention scores (without position info)",
                     color=GREY).scale(0.45).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


class RoPEIntro(Scene):
    def construct(self):
        header = Text("Solution: Rotary Position Embedding (RoPE)", color=BW, weight=BOLD).scale(0.8).to_edge(UP)
        self.play(Write(header))

        idea = Text("Idea: rotate each vector by an angle that depends on its position",
                     color=GREY).scale(0.5).shift(UP * 2.0)
        self.play(FadeIn(idea))

        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": GREY, "stroke_opacity": 0.3}
        ).scale(0.9)
        self.play(Create(plane))

        vec = Vector([2, 0], color=BW)
        self.play(GrowArrow(vec))

        angle_tracker = ValueTracker(0)

        def update_vec(v):
            theta = angle_tracker.get_value()
            v.become(Vector([2 * np.cos(theta), 2 * np.sin(theta)], color=BW))

        vec.add_updater(update_vec)

        angle_label = always_redraw(lambda: MathTex(
            r"\theta \cdot pos", color=BW
        ).scale(0.6).next_to(vec, UR, buff=0.2))

        self.add(angle_label)
        self.play(angle_tracker.animate.set_value(PI / 2), run_time=2)
        self.wait()
        vec.clear_updaters()

        note = Text("Each position gets its own rotation angle", color=GREY).scale(0.45).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class RoPEMath(Scene):
    def construct(self):
        header = Text("RoPE: the rotation matrix", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        matrix = MathTex(
            r"R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}",
            color=BW
        ).scale(0.9)
        self.play(Write(matrix))
        self.wait()

        apply_q = MathTex(r"q' = R(m\theta)\, q", color=BW).scale(0.9).shift(DOWN * 1.2)
        apply_k = MathTex(r"k' = R(n\theta)\, k", color=BW).scale(0.9).shift(DOWN * 2.0)
        self.play(Write(apply_q))
        self.play(Write(apply_k))

        note = Text("m, n are token positions — each pair of dims gets rotated",
                     color=GREY).scale(0.45).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


class RelativePositionMagic(Scene):
    def construct(self):
        header = Text("The key trick: relative position falls out naturally", color=BW, weight=BOLD)
        header.scale(0.7).to_edge(UP)
        self.play(Write(header))

        derivation = MathTex(
            r"q'^T k' &= (R(m\theta) q)^T (R(n\theta) k) \\",
            r"&= q^T R(m\theta)^T R(n\theta)\, k \\",
            r"&= q^T R((n-m)\theta)\, k",
            color=BW
        ).scale(0.75)
        self.play(Write(derivation[0]))
        self.wait()
        self.play(Write(derivation[1]))
        self.wait()
        self.play(Write(derivation[2]))

        box = SurroundingRectangle(derivation[2], color=BW)
        self.play(Create(box))

        note = Text("Dot product only depends on (n - m): the relative distance!",
                     color=GREY).scale(0.45).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


class RotationVisual(Scene):
    def construct(self):
        header = Text("Visualizing relative rotation", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": GREY, "stroke_opacity": 0.3}
        ).scale(0.9)
        self.play(Create(plane))

        q_vec = Vector([2, 0], color=BW)
        k_vec = Vector([2, 0], color=GREY)
        q_label = Text("q (pos m)", color=BW).scale(0.4).next_to(q_vec, UP)
        k_label = Text("k (pos n)", color=GREY).scale(0.4).next_to(k_vec, DOWN)

        self.play(GrowArrow(q_vec), GrowArrow(k_vec), FadeIn(q_label), FadeIn(k_label))

        m_tracker = ValueTracker(0)
        n_tracker = ValueTracker(0)
        theta = 0.6

        def update_q(v):
            v.become(Vector([2 * np.cos(m_tracker.get_value() * theta),
                              2 * np.sin(m_tracker.get_value() * theta)], color=BW))

        def update_k(v):
            v.become(Vector([2 * np.cos(n_tracker.get_value() * theta),
                              2 * np.sin(n_tracker.get_value() * theta)], color=GREY))

        q_vec.add_updater(update_q)
        k_vec.add_updater(update_k)
        q_label.add_updater(lambda l: l.next_to(q_vec, UP))
        k_label.add_updater(lambda l: l.next_to(k_vec, DOWN))

        self.play(m_tracker.animate.set_value(2), n_tracker.animate.set_value(5), run_time=3)
        self.wait()

        note = Text("Both rotate, but the ANGLE BETWEEN them stays tied to (n - m)",
                     color=GREY).scale(0.45).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(3)

        q_vec.clear_updaters()
        k_vec.clear_updaters()
        q_label.clear_updaters()
        k_label.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects])


class WhyItWorks(Scene):
    def construct(self):
        header = Text("Why RoPE is elegant", color=BW, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        points = VGroup(
            Text("• Encodes position without extra learned embeddings", color=BW),
            Text("• Naturally generalizes to longer sequences", color=BW),
            Text("• Preserves relative distance in the dot product", color=BW),
            Text("• Used in LLaMA, GPT-NeoX, and many modern LLMs", color=BW),
        ).scale(0.55).arrange(DOWN, aligned_edge=LEFT, buff=0.4)

        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.3) for p in points], lag_ratio=0.3))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


class Recap(Scene):
    def construct(self):
        header = Text("Recap", color=BW, weight=BOLD).scale(1.2)
        self.play(Write(header))
        self.play(header.animate.to_edge(UP))

        steps = VGroup(
            Text("1. Tokens → vectors (Q, K, V)", color=BW),
            Text("2. Attention = softmax(QKᵀ/√d) · V", color=BW),
            Text("3. Problem: no notion of position", color=BW),
            Text("4. RoPE rotates Q and K by position-dependent angles", color=BW),
            Text("5. Result: attention naturally depends on relative position", color=BW),
        ).scale(0.55).arrange(DOWN, aligned_edge=LEFT, buff=0.35)

        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in steps], lag_ratio=0.25))
        self.wait(3)

        outro = Text("Thanks for watching.", color=BW).scale(0.9)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.play(Write(outro))
        self.wait(2)
        self.play(FadeOut(outro))
