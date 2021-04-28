"""
Apply codegen to rs backend.

The idea is that when there are any changes in wgpu.h that affect how rs.py
should be written, this module will:

* For enums: automatically update the mappings.
* For flags: report discrepancies.
* For structs and functions: update the code, so a diff of rs.py quickly
  shows if manual changes are needed.

Note that the apipatcher will also patch rs.py, but where that codegen
focuses on the API, here we focus on the C library usage.
"""

import os

from codegen.utils import print, lib_dir, blacken, Patcher
from codegen.hparser import get_h_parser
from codegen.idlparser import get_idl_parser


mappings_preamble = '''
""" Mappings for the rs backend. """

# THIS CODE IS AUTOGENERATED - DO NOT EDIT

# flake8: noqa
'''.lstrip()


def compare_flags():
    """For each flag in WebGPU:

    * Verify that there is a corresponding flag in wgpu.h
    * Verify that all fields are present too.
    * Verify that the (integer) value is equal.

    Verification fails lead to prints
    TODO: should also end up in codegen report.
    """

    idl = get_idl_parser()
    hp = get_h_parser()

    for name, flag in idl.flags.items():
        if name not in hp.flags:
            print(f"Flag {name} missing in wgpu.h")
        else:
            for key, val in flag.items():
                if key not in hp.flags[name]:
                    print(f"Flag field {name}.{key} missing in wgpu.h")
                elif val != hp.flags[name][key]:
                    print(f"Warning: Flag field {name}.{key} have different values.")


def write_mappings():
    """Generate the file with dicts to map enums strings to ints. This
    also compares the enums in wgpu-native with WebGPU, and reports any
    missing ones.
    """

    idl = get_idl_parser()
    hp = get_h_parser()

    # Init generated code
    pylines = [mappings_preamble]

    # Create enummap, which allows the rs backend to resolve enum field names
    # to the corresponding integer value.
    enummap = {}
    for name in idl.enums:
        if name not in hp.enums:
            print(f"Enum {name} missing in wgpu.h")
            continue
        for ikey in idl.enums[name].values():
            hkey = ikey
            hkey = hkey.replace("1d", "D1").replace("2d", "D2").replace("3d", "D3")
            if "index" not in name.lower():
                # Yuk! but wgpu.h will probably align to WebGPU soon, so should be temporary
                hkey = (
                    hkey.replace("uint8", "Uchar")
                    .replace("uint16", "Ushort")
                    .replace("uint32", "Uint")
                )
                hkey = (
                    hkey.replace("sint8", "Char")
                    .replace("sint16", "Short")
                    .replace("sint32", "Int")
                )
                hkey = hkey.replace("float16", "Half").replace("float32", "Float")
                hkey = hkey.replace("x2", "2").replace("x3", "3").replace("x4", "4")
            hkey = hkey.replace("-", " ").title().replace(" ", "")
            if hkey in hp.enums[name]:
                enummap[name + "." + ikey] = hp.enums[name][hkey]
            else:
                print(f"Enum field {name}.{ikey} missing in wgpu.h")

    # Write enummap
    pylines.append(f"# There are {len(enummap)} enum mappings\n")
    pylines.append("enummap = {")
    for key in sorted(enummap.keys()):
        pylines.append(f'    "{key}": {enummap[key]!r},')
    pylines.append("}\n")

    # Some structs have fields that are enum values. The rs backend
    # must be able to resolve these too.
    cstructfield2enum = {}
    for structname, struct in hp.structs.items():
        for key, val in struct.items():
            if isinstance(val, str) and val.startswith("WGPU"):
                enumname = val[4:].split("/")[0]
                if enumname in idl.enums:
                    cstructfield2enum[f"{structname[4:]}.{key}"] = enumname
                else:
                    pass  # a struct

    # Write cstructfield2enum
    pylines.append(f"# There are {len(cstructfield2enum)} struct-field enum mappings\n")
    pylines.append("cstructfield2enum = {")
    for key in sorted(cstructfield2enum.keys()):
        pylines.append(f'    "{key}": {cstructfield2enum[key]!r},')
    pylines.append("}\n")

    # We need to resolve feature flags into names
    features_names = {}
    for name, val in hp.flags["Features"].items():
        features_names[val] = name.lower()
    pylines.append(f"# Inverse flag map for features\n")
    pylines.append("feature_names = {")
    for key in sorted(features_names.keys()):
        pylines.append(f"    {key}: {features_names[key]!r},")
    pylines.append("}\n")

    # Wrap up
    code = blacken("\n".join(pylines))  # just in case; code is already black
    with open(os.path.join(lib_dir, "backends", "rs_mappings.py"), "wb") as f:
        f.write(code.encode())
    print(
        f"Wrote {len(enummap)} enum mappings and {len(cstructfield2enum)} struct-field mappings to rs_mappings.py"
    )


def patch_rs_backend(code):
    """Given the Python code, applies patches to annotate functions
    calls and struct instantiations.

    For functions:

    * Verify that the function exists in wgpu.h. If not, add a fixme comment.
    * Add a comment showing correspinding signature from wgpu.h.

    For structs:

    * Verify that the struct name exists.
    * Verify that the correct form (pointer or not) is used.
    * Verify that all used fields exists.
    * Annotate any missing fields.
    * Add a comment that shows all fields and their type.

    """

    for patcher in [CommentRemover(), FunctionPatcher(), StructPatcher()]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


class CommentRemover(Patcher):

    triggers = "# FIXME: unknown C", "# FIXME: invalid C", "# H:"

    def apply(self, code):
        self._init(code)
        for line, i in self.iter_lines():
            if line.lstrip().startswith(self.triggers):
                self.remove_line(i)


class FunctionPatcher(Patcher):
    def apply(self, code):
        self._init(code)
        hp = get_h_parser()
        count = 0
        detected = set()

        for line, i in self.iter_lines():
            if "lib.wgpu_" in line:
                start = line.index("lib.wgpu_") + 4
                end = line.index("(", start)
                name = line[start:end]
                indent = " " * (len(line) - len(line.lstrip()))
                if name not in hp.functions:
                    msg = f"unknown C function {name}"
                    self.insert_line(i, f"{indent}# FIXME: {msg}")
                    print(f"ERROR: {msg}")
                else:
                    detected.add(name)
                    anno = hp.functions[name].replace(name, "f").strip(";")
                    self.insert_line(i, indent + f"# H: " + anno)
                    count += 1

        print(f"Validated {count} C function calls")

        # Determine what functions were not detected
        # There are still quite a few, so we don't list them yet
        ignore = (
            "wgpu_create_surface_from",
            "wgpu_set_log_level",
            "wgpu_get_version",
            "wgpu_set_log_callback",
        )
        unused = set(name for name in hp.functions if not name.startswith(ignore))
        unused.difference_update(detected)
        print(f"Not using {len(unused)} C functions")


class StructPatcher(Patcher):
    def apply(self, code):
        self._init(code)
        hp = get_h_parser()

        count = 0
        line_index = -1
        brace_depth = 0

        for line, i in self.iter_lines():

            if "new_struct_p(" in line or "new_struct(" in line:
                if line.lstrip().startswith("def "):
                    continue  # Implementation
                if "_new_struct" in line:
                    continue  # Implementation
                if "new_struct_p()" in line or "new_struct()" in line:
                    continue  # Comments or docs
                line_index = i
                j = line.index("new_struct")
                line = line[j:]  # start brace searching from right pos
                brace_depth = 0

            if line_index >= 0:
                for c in line:
                    if c == "#":
                        break
                    elif c == "(":
                        brace_depth += 1
                    elif c == ")":
                        brace_depth -= 1
                        assert brace_depth >= 0
                        if brace_depth == 0:
                            self._validate_struct(hp, line_index, i)
                            count += 1
                            line_index = -1
                            break

        print(f"Validated {count} C structs")

    def _validate_struct(self, hp, i1, i2):
        """Validate a specific struct usage."""

        lines = self.lines[
            i1 : i2 + 1
        ]  # note: i2 is the line index where the closing brace is
        indent = " " * (len(lines[-1]) - len(lines[-1].lstrip()))

        if len(lines) == 1:
            # Single line - add a comma before the closing brace
            print(
                "Notice: made a struct multiline. Rerun codegen to validate the struct."
            )
            line = lines[0]
            i = line.rindex(")")
            line = line[:i] + "," + line[i:]
            self.replace_line(i1, line)
            return
        elif len(lines) == 3 and lines[1].count("="):
            # Triplet - add a comma after the last element
            print(
                "Notice: made a struct multiline. Rerun codegen to validate the struct."
            )
            self.replace_line(i1 + 1, self.lines[i1 + 1] + ",")
            return

        # We can assume that the struct is multi-line and formatted by Black!
        assert len(lines) >= 3

        # Get struct name, and verify
        name = lines[1].strip().strip(',"')
        struct_name = name.strip(" *")
        if name.endswith("*"):
            if "new_struct_p" not in lines[0]:
                self.insert_line(
                    i1, indent + f"# FIXME: invalid C struct, use new_struct_p()"
                )
        else:
            if "new_struct_p" in lines[0]:
                self.insert_line(
                    i1, indent + f"# FIXME: invalid C struct, use new_struct()"
                )

        # Get struct object and create annotation line
        if struct_name not in hp.structs:
            msg = f"unknown C struct {struct_name}"
            self.insert_line(i1, f"{indent}# FIXME: {msg}")
            print(f"ERROR: {msg}")
            return
        else:
            struct = hp.structs[struct_name]
            fields = ", ".join(f"{key}: {val}" for key, val in struct.items())
            self.insert_line(i1, indent + f"# H: " + fields)

        # Check keys
        keys_found = []
        for j in range(2, len(lines) - 1):
            line = lines[j]
            key = line.split("=")[0].strip()
            if key.startswith("# not used:"):
                key = key.split(":")[1].split("=")[0].strip()
            elif key.startswith("#"):
                continue
            keys_found.append(key)
            if key not in struct:
                msg = f"unknown C struct field {struct_name}.{key}"
                self.insert_line(i1 + j, f"{indent}# FIXME: {msg}")
                print(f"ERROR: {msg}")

        # Insert comments for unused keys
        more_lines = []
        for key in struct:
            if key not in keys_found:
                more_lines.append(indent + f"    # not used: {key}")
        if more_lines:
            self.insert_line(i2, "\n".join(more_lines))
