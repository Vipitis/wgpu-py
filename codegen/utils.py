"""
Codegen utils.
"""

import os
import tempfile

import black


lib_dir = os.path.abspath(os.path.join(__file__, "..", "..", "wgpu"))


def to_snake_case(name):
    """Convert a name from camelCase to snake_case. Names that already are
    snake_case remain the same.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0 and name2[-1] != "_":
            name2 += "_"
        name2 += c2
    return name2


def to_camel_case(name):
    """Convert a name from snake_case to camelCase. Names that already are
    camelCase remain the same.
    """
    is_capital = False
    name2 = ""
    for c in name:
        if c == "_" and name2:
            is_capital = True
        elif is_capital:
            name2 += c.upper()
            is_capital = False
        else:
            name2 += c
    return name2


def blacken(src, singleline=False):
    """Format the given src string using black. If singleline is True,
    all function signatures become single-line, so they can be parsed
    and updated.
    """
    # Normal black
    result = black.format_str(src, mode=black.FileMode())

    # Make defs single-line. You'd think that setting the line length
    # to a very high number would do the trick, but it does not.
    if singleline:
        lines1 = result.splitlines()
        lines2 = []
        in_sig = False
        comment = ""
        for line in lines1:
            if in_sig:
                # Handle comment
                line, _, c = line.partition("#")
                line = line.rstrip()
                c = c.strip()
                if c:
                    comment += " " + c.strip()
                # Detect end
                if line.endswith("):"):
                    in_sig = False
                # Compose line
                current_line = lines2[-1]
                if not current_line.endswith("("):
                    current_line += " "
                current_line += line.lstrip()
                # Finalize
                if not in_sig:
                    # Remove trailing spaces and commas
                    current_line = current_line.replace(" ):", "):")
                    current_line = current_line.replace(",):", "):")
                    # Add comment
                    if comment:
                        current_line += "  #" + comment
                        comment = ""
                lines2[-1] = current_line
            else:
                lines2.append(line)
                line_nc = line.split("#")[0].strip()
                if (
                    line_nc.startswith(("def ", "async def", "class "))
                    and "(" in line_nc
                ):
                    if not line_nc.endswith("):"):
                        in_sig = True
        lines2.append("")
        result = "\n".join(lines2)

    return result


class Patcher:
    """Class to help patch a Python module. Supports iterating (over
    lines, classes, properties, methods), and applying diffs (replace,
    remove, insert).
    """

    def __init__(self, code=None):
        self._init(code)

    def _init(self, code):
        """Subclasses can call this to reset the patcher."""
        self.lines = []
        self._diffs = {}
        self._classes = {}
        if code:
            self.lines = blacken(code, True).splitlines()  # inf line length

    def remove_line(self, i):
        """Remove the line at the given position. There must not have been
        an action on line i.
        """
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "remove"

    def insert_line(self, i, line):
        """Insert a new line at the given position. It's ok if there
        has already been an insertion an line i, but there must not have been
        any other actions.
        """
        if i in self._diffs and self._diffs[i][1] == "insert":
            cur_line = self._diffs[i][2]
            self._diffs[i] = i, "insert", cur_line + "\n" + line
        else:
            assert i not in self._diffs, f"Line {i} already has a diff"
            self._diffs[i] = i, "insert", line

    def replace_line(self, i, line):
        """Replace the line at the given position with another line.
        There must not have been an action on line i.
        """
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "replace", line

    def dumps(self, format=True):
        """Return the patched result as a string."""
        lines = self.lines.copy()
        # Apply diff
        diffs = sorted(self._diffs.values())
        for diff in reversed(diffs):
            if diff[1] == "remove":
                lines.pop(diff[0])
            elif diff[1] == "insert":
                lines.insert(diff[0], diff[2])
            elif diff[1] == "replace":
                lines[diff[0]] = diff[2]
            else:  # pragma: no cover
                raise ValueError(f"Unknown diff: {diff}")
        # Format
        text = "\n".join(lines)
        if format:
            try:
                text = blacken(text)
            except black.InvalidInput as err:  # pragma: no cover
                # If you get this error, it really helps to load the code
                # in an IDE to see where the error is. Let's help with that ...
                filename = os.path.join(tempfile.gettempdir(), "wgpu_patcher_fail.py")
                with open(filename, "wb") as f:
                    f.write(text.encode())
                err = str(err)
                err = err if len(err) < 78 else err[:77] + "…"
                raise RuntimeError(
                    f"It appears that the patcher has generated invalid Python:"
                    f"\n\n    {err}\n\n"
                    f'Wrote the generated (but unblackened) code to:\n\n  "{filename}"'
                )

        return text

    def iter_lines(self, start_line=0):
        """Generator to iterate over the lines.
        Each iteration yields (line, linenr)
        """
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            yield line, i

    def iter_classes(self, start_line=0):
        """Generator to iterate over the classes.
        Each iteration yields (classname, linenr_start, linenr_end),
        where linenr_end is the last line of code.
        """
        current_class = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if current_class and sline:
                if sline.startswith("    "):
                    current_class[2] = i
                else:  # code has less indentation -> something new
                    yield current_class
                    current_class = None
            if line.startswith("class "):
                name = line.split(":")[0].split("(")[0].split()[-1]
                current_class = [name, i, i]
        if current_class:
            yield current_class

    def iter_properties(self, start_line=0):
        """Generator to iterate over the properties.
        Each iteration yields (classname, linenr_start, linenr_end),
        where linenr_start is the line that says `@property`,
        and linenr_end is the last line of code.
        """
        return self._iter_props_and_methods(start_line, True)

    def iter_methods(self, start_line=0):
        """Generator to iterate over the methods.
        Each iteration yields (classname, linenr_start, linenr_end)
        where linenr_end is the last line of code.
        """
        return self._iter_props_and_methods(start_line, False)

    def _iter_props_and_methods(self, start_line, find_props):
        prop_mark = None
        current_def = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if current_def and sline:
                if sline.startswith("        "):
                    current_def[2] = i
                else:
                    yield current_def
                    current_def = None
            if sline and not sline.startswith("    "):
                break  # exit class
            if line.startswith(("    def ", "    async def ")):
                name = line.split("(")[0].split()[-1]
                if prop_mark and find_props:
                    current_def = [name, prop_mark, i]
                elif not prop_mark and not find_props:
                    current_def = [name, i, i]
            if line.startswith("    @property"):
                prop_mark = i
            elif sline and not sline.startswith("#"):
                prop_mark = None

        if current_def:
            yield current_def