# -*- coding: utf-8 -*-
"""Генератор SVG-схемы агентов. Правим здесь, перезапускаем, схема обновляется."""
import html

W = 1000
PAD = 14
LH = 17
HEADER_H = 30
LBL_H = 15

C = dict(data="#4C7A87", qual="#0F6E64", rel="#A24A2E", opt="#3A5590",
         blend="#6B4C87", orc="#B8791A", veto="#9E2B25", mute="#647169",
         ink="#16211F", soft="#3E4A47", border="#D9DFD9", surf="#FFFFFF",
         surf2="#EEF1EE", paper="#F5F7F4")

SANS = "'IBM Plex Sans','Segoe UI',Arial,sans-serif"
MONO = "'IBM Plex Mono','Consolas',monospace"

out = []


def esc(s):
    return html.escape(s, quote=False)


def node_height(nin, nout, nrule=0):
    h = HEADER_H
    for n in (nin, nout):
        if n:
            h += LBL_H + n * LH + 10
    if nrule:
        h += LBL_H + nrule * LH + 10
    return h


def node(x, y, w, title, color, ins, outs, veto=False, rule=None,
         rule_lbl="правило"):
    h = node_height(len(ins), len(outs), len(rule) if rule else 0)
    out.append('<rect x="%s" y="%s" width="%s" height="%s" rx="10" fill="%s" '
               'stroke="%s" stroke-width="1"/>' % (x, y, w, h, C["surf"], C["border"]))
    out.append('<rect x="%s" y="%s" width="%s" height="3" rx="1.5" fill="%s"/>'
               % (x, y, w, color))
    cy = y + 21
    out.append('<text x="%s" y="%s" font-family="%s" font-size="13.5" font-weight="700" '
               'fill="%s">%s</text>' % (x + PAD, cy, SANS, color, esc(title)))
    if veto:
        bx = x + PAD + len(title) * 8.2 + 12
        out.append('<rect x="%s" y="%s" width="44" height="15" rx="4" fill="#F7E3E0"/>'
                   % (bx, cy - 11))
        out.append('<text x="%s" y="%s" font-family="%s" font-size="9" font-weight="600" '
                   'fill="%s" text-anchor="middle">ВЕТО</text>'
                   % (bx + 22, cy - 0.5, MONO, C["veto"]))
    cy = y + HEADER_H

    def section(lines, label, shaded):
        nonlocal cy
        sh = LBL_H + len(lines) * LH + 10
        if shaded:
            out.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>'
                       % (x + 1, cy, w - 2, sh, C["surf2"]))
        out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s"/>'
                   % (x, cy, x + w, cy, C["border"]))
        out.append('<text x="%s" y="%s" font-family="%s" font-size="8.5" letter-spacing="1" '
                   'fill="%s">%s</text>'
                   % (x + PAD, cy + 12, MONO, C["mute"], esc(label.upper())))
        ty = cy + LBL_H + 12
        for text, mono in lines:
            font = MONO if mono else SANS
            size = "11" if mono else "11.5"
            out.append('<text x="%s" y="%s" font-family="%s" font-size="%s" fill="%s">%s</text>'
                       % (x + PAD, ty, font, size, C["soft"], esc(text)))
            ty += LH
        cy += sh

    if ins:
        section(ins, "вход", True)
    if rule:
        section(rule, rule_lbl, False)
    if outs:
        section(outs, "выход", False)
    return h


def arrow(x, y, label=None):
    h = 34 if label else 24
    out.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#BFC8BF" stroke-width="2"/>'
               % (x, y, x, y + h - 8))
    out.append('<path d="M%s %s L%s %s L%s %s Z" fill="#BFC8BF"/>'
               % (x - 5, y + h - 8, x + 5, y + h - 8, x, y + h))
    if label:
        tw = len(label) * 6.0 + 18
        out.append('<rect x="%s" y="%s" width="%s" height="16" rx="5" fill="%s" stroke="%s"/>'
                   % (x - tw / 2, y + 6, tw, C["surf2"], C["border"]))
        out.append('<text x="%s" y="%s" font-family="%s" font-size="9.5" fill="%s" '
                   'text-anchor="middle">%s</text>'
                   % (x, y + 17.5, MONO, C["mute"], esc(label)))
    return h


# ------------------------------- сборка -------------------------------
y = 20
out.append('<text x="%s" y="%s" font-family="%s" font-size="19" font-weight="700" fill="%s" '
           'text-anchor="middle">Граф агентов: вход и выход каждого узла</text>'
           % (W / 2, y + 18, SANS, C["ink"]))
y += 30
out.append('<text x="%s" y="%s" font-family="%s" font-size="11.5" fill="%s" '
           'text-anchor="middle">ВЕТО — право запретить действие независимо от выгоды. '
           'Теги указаны с префиксом установки.</text>' % (W / 2, y + 12, SANS, C["mute"]))
y += 34

sw = (W - 80 - 3 * 12) / 4
srcs = [("avt_tags.csv", "71 тег"), ("242000_tags.csv", "26 тегов"),
        ("ЛИМС", "6 точек отбора"), ("ПАК", "сера, плотность")]
for i, (a, b) in enumerate(srcs):
    sx = 40 + i * (sw + 12)
    out.append('<rect x="%s" y="%s" width="%s" height="42" rx="8" fill="%s" stroke="%s"/>'
               % (sx, y, sw, C["surf2"], C["border"]))
    out.append('<text x="%s" y="%s" font-family="%s" font-size="10.5" fill="%s" '
               'text-anchor="middle">%s</text>' % (sx + sw / 2, y + 18, MONO, C["ink"], esc(a)))
    out.append('<text x="%s" y="%s" font-family="%s" font-size="10" fill="%s" '
               'text-anchor="middle">%s</text>' % (sx + sw / 2, y + 32, SANS, C["mute"], esc(b)))
y += 42
y += arrow(W / 2, y)

y += node(40, y, W - 80, "АГЕНТ ДАННЫХ", C["data"],
          [("avt_tags.csv · 242000_tags.csv · ЛИМС 6 точек · ПАК 2 сигнала", True),
           ("чистит заглушки 307 и 251, ловит остановки и залипания", False),
           ("считает возраст проб с учётом задержки публикации ЛИМС 4 ч", False)],
          [("kip             значения тегов          HT_T5: 370.3 ...", True),
           ("quality         значение + источник + возраст   9.8, ПАК, 0.2ч", True),
           ("overall         вердикт                 ok / degraded / insufficient", True),
           ("notes           причины                 ЛИМС устарел на 26ч", True)],
          veto=True)
y += arrow(W / 2, y, "ProcessState")

colw = (W - 80 - 20) / 2
h1 = node(40, y, colw, "АГЕНТ КАЧЕСТВА", C["qual"],
          [("HT:T5 T6 T11 F15 P13 F26", True),
           ("AVT:T66 F28 F30 F32 T33 T20", True),
           ("ЛИМС сера и Т95 · ПАК сера", False)],
          [("sulfur_now      9.8 мг/кг", True),
           ("sulfur_3h      10.2 мг/кг", True),
           ("t95_now         349 °C", True),
           ("confidence      0.72", True)])
h2 = node(40 + colw + 20, y, colw, "АГЕНТ НАДЁЖНОСТИ", C["rel"],
          [("HT:W10 — перепад давления", True),
           ("HT:T5 T6 T11 — температуры слоя", True),
           ("HT:F2 P24 — водород", True)],
          [("temp_30d        369 °C", True),
           ("headroom        33 °C до предела", True),
           ("max_temp        378 °C — рамка", True),
           ("regime_allowed  true / false", True)],
          veto=True)
y += max(h1, h2)
y += arrow(W / 2, y, "прогноз качества · предел по температуре")

y += node(40, y, W - 80, "АГЕНТ БЛЕНДИНГА", C["blend"],
          [("профили резервуаров — фактические квантили качества за 3 года", False),
           ("спецификация — ВХОДНОЙ ПАРАМЕТР: сера, Т95, цетановое число", False),
           ("прогноз качества от агента качества", False)],
          [("shippable_range до 15.8 мг/кг — в каком качестве отгружаем", True),
           ("recipe          current 50% · deep 40% · normal 10%", True),
           ("improver_pct    0 %   при пределе 3 %", True),
           ("cost            1.024 усл.ед./т", True)],
          veto=True)
y += arrow(W / 2, y, "рамка отгружаемого качества + цена")

y += node(40, y, W - 80, "АГЕНТ ОПТИМИЗАЦИИ", C["opt"],
          [("рычаги: HT:T5 F26 F15 P13 F2 · AVT:T33 T20", True),
           ("от качества — во что превратится сера", False),
           ("от надёжности — предел по температуре", False),
           ("от блендинга — рамка качества и цена", False)],
          [("scenarios       шаги ±1 % и ±2 %, включая ничего не менять", True),
           ("changes         T5: 370.3 -> 377.7", True),
           ("rejected_by     T5 выше предела 378", True),
           ("best_id         T5+2%     feasible: true", True)])
y += arrow(W / 2, y, "сценарии с метриками")

y += node(40, y, W - 80, "ОРКЕСТРАТОР", C["orc"],
          [("вердикт данных · прогноз качества · ограничения надёжности", False),
           ("сценарии оптимизации · рецептура и цена блендинга", False)],
          [("status          recommendation / no_action / refusal", True),
           ("семь блоков оператору — см. ниже", True),
           ("agent_verdicts  все вердикты целиком, для дашборда", True)],
          rule=[("1 · данные непригодны       -> отказ", True),
                ("2 · надёжность против      -> отказ", True),
                ("3 · блендинг не сводит     -> отказ", True),
                ("4 · проблемы нет           -> вмешательство не требуется", True),
                ("5 · иначе                  -> самый дешёвый из допустимых", True)],
          rule_lbl="правило разрешения конфликта")
y += arrow(W / 2, y)

blocks = [("1", "Время и состояние", "15.06.2024 12:00 · сера 9.8 мг/кг (ПАК) · ЛИМС 26 ч"),
          ("2", "Проблема", "запас до предела отгрузки 0.4 мг/кг"),
          ("3", "Действие", "HT:T5   370.3 -> 377.7 °C   (+2 %)"),
          ("4", "Эффект", "сера 8.2 · смесь 9.6 · цена 1.024 · катализатор -1 °C"),
          ("5", "Ограничения", "сера · Т95 · цетан · доли 100 % · предел 378 — все пройдены"),
          ("6", "Уверенность", "средняя — лабораторный анализ устарел"),
          ("7", "Объяснение", "из 35 вариантов 28 отброшено, выбран самый дешёвый")]
bh = LBL_H + len(blocks) * LH + 18
out.append('<rect x="40" y="%s" width="%s" height="%s" rx="10" fill="%s" stroke="%s" '
           'stroke-width="1.5"/>' % (y, W - 80, bh, C["surf2"], C["orc"]))
out.append('<text x="%s" y="%s" font-family="%s" font-size="8.5" letter-spacing="1" '
           'fill="%s">ТИПОВОЙ ВЫХОД ОПЕРАТОРУ — ПРИМЕР</text>'
           % (40 + PAD, y + 15, MONO, C["mute"]))
ty = y + LBL_H + 14
for n, k, v in blocks:
    out.append('<text x="%s" y="%s" font-family="%s" font-size="10" fill="%s">%s</text>'
               % (40 + PAD, ty, MONO, C["orc"], n))
    out.append('<text x="%s" y="%s" font-family="%s" font-size="11" font-weight="600" '
               'fill="%s">%s</text>' % (40 + PAD + 16, ty, SANS, C["ink"], esc(k)))
    out.append('<text x="%s" y="%s" font-family="%s" font-size="11" fill="%s">%s</text>'
               % (40 + PAD + 165, ty, SANS, C["soft"], esc(v)))
    ty += LH
y += bh + 24

H = int(y)
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s">\n'
       '<rect width="%s" height="%s" fill="%s"/>\n' % (W, H, W, H, W, H, C["paper"])
       + "\n".join(out) + "\n</svg>\n")

with open("agents_graph.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("agents_graph.svg готов: %s x %s" % (W, H))
