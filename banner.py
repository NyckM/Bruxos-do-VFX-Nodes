# -*- coding: utf-8 -*-
"""Banner de inicializacao do Bruxos do VFX (logo em ASCII, verde e roxo)."""
import os
import sys

VERDE = (34, 197, 94)
ROXO = (168, 85, 247)
VERDE2 = (74, 222, 128)

_ART = [
    '                   :          -=',
    '                  :@-        =@=     -.',
    '               -==+@*==-    .%@+   -#@+',
    '               :::=@+:::   ==%@* .#@@@%.',
    '                  :%:     +@%#@# .%@@@@+',
    '                   .      *@@@@@- =@@@@%.',
    '                        *=:#@@@@%= +@@@@=',
    '                       +@@*-*@@@@@: #@@@%.   .::.',
    '                      :@@@@@#*#%@@# .%@@@=   :*#.',
    ' -*##*=          .=.  .#@@@@@@@%%%@= -@@@%.   .--:',
    '*@@@@@@#       -*%+  ++:+@@@@@@@@@@@- =@@@=   .=#-',
    '%@@@@@@%    .=#@@@:  %@%+=+%@@@@@@@@%: *@@%     :+*',
    ':*%@@%*:  :+%@@@@#   *@@@@#***##%%@@@# .#@@=     .:',
    '   ... .-#@@@@@@#.    +#%@@@@@@@@@%#+-   .::',
    '     :+%@@@@@@@%:       .:--=+%**%*     .',
    '   -*@@@@@@@@#=.  :*+-:.     .#=-#%::-+*####******###-',
    '.=%@@@@@@%#=:  :=#@@@@@%#**+++*%###***#%%%#%@@@@@@@*:',
    '.-+*%%#+-.  :+#@@@@@@@@@%##********###**#%@@%#%%%+:',
    '     .  :=*%@@@@@@@@@%%%%%####**++**##%%##*###+:',
    '   .:=*%@@@@@@@@@@@@@%%%###########*****#*+-:.:',
    ' .=#%@@@@@@@%%#####******+**+***####*+=-::-+#%@-',
    '    .:-=+*##%%%%%%%%%%%%####*++=---==+*#%@@@@@@#',
    '              .........       :+#%@@@@@@@@@@@@@@-',
    '                          :.    .-=*#%@@@@@@@@@@#',
    '                         :@*         .:=+#%@@@@@@-',
    '                      ...=@*....           :-+*%@#',
    '                     :***#@%***+                .:',
    '                         :@*',
    '                         :@*',
]

_MAXW = max((len(l) for l in _ART), default=1)


def _enable_ansi():
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            try:
                os.system("")
            except Exception:
                return False
    return True


def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return True


def _rgb(r, g, b):
    return "\033[38;2;{};{};{}m".format(r, g, b)


def _blend(t):
    r = int(VERDE[0] + (ROXO[0] - VERDE[0]) * t)
    g = int(VERDE[1] + (ROXO[1] - VERDE[1]) * t)
    b = int(VERDE[2] + (ROXO[2] - VERDE[2]) * t)
    return _rgb(r, g, b)


def render(node_count=None, version=""):
    color = _supports_color() and _enable_ansi()
    reset = "\033[0m" if color else ""
    dim = "\033[2m" if color else ""
    out = ["", ""]
    for y, line in enumerate(_ART):
        if not color:
            out.append("  " + line)
            continue
        buf = "  "
        for x, c in enumerate(line):
            if c == " ":
                buf += " "
            else:
                t = (x / _MAXW) * 0.6 + (y / len(_ART)) * 0.4
                buf += _blend(min(1.0, t)) + c
        out.append(buf + reset)
    g = _rgb(*VERDE2) if color else ""
    p = _rgb(*ROXO) if color else ""
    ver = (" v" + version) if version else ""
    nc = ("  -  {} nodes".format(node_count)) if node_count is not None else ""
    out.append("")
    out.append("  " + g + "B R U X O S" + reset + p + "  D O   V F X" + reset
               + dim + ver + nc + reset)
    out.append("  " + dim + "remocao de objetos - upscale - face swap - Bernini/Wan" + reset)
    out.append("")
    return "\n".join(out)


def print_banner(node_count=None, version=""):
    try:
        print(render(node_count, version), flush=True)
    except Exception:
        pass
