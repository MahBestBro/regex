# zig-regex
 A single file regex library written in and for Zig.

 **Note:** This library is still in development.  For now I would recommend using another library if you are looking for something robust.  
 
 This library was mainly inspired by the rough implementation of Ken Thompson's algorithm outlined in [this article](https://swtch.com/~rsc/regexp/regexp1.html) by Russ Cox, though not everything is based off it.

## Getting Started
 Just download the `regex.zig` file and include it into your project however you want.  

## Example Usage
```zig
const std = @import("std");
const regex = @import("regex.zig");

pub fn main() !void
{
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const rx = try regex.Regex.compile("ab*c", arena.allocator());
    defer rx.deinit();
    
    if (rx.match("abc")) 
    {
        std.debug.print("Hooray!\n", .{});
    } 
    else 
    {
        std.debug.print("Uh oh...\n", .{});
    }
}
```

## Notation
 Like all regular expressions, non-operator characters which are next to each other concatenate (so `abc` would match "abc"). Brackets are denoted by normal parenthesis (i.e., `()`).

 The following regex operators are supported as of current:
 * `|` - If `R` and `S` are regular expressions, then `R|S` matches `R` or `S` (Note: This has the highest precedence, so `a|bc*` is equivalent to `a|(bc*)`, not `(a|b)c*`).
 * `*` - If `R` is a regular expression, then `R*` matches 0 or more repetitions of `R`.
 * `+` - Same as `*` but 1 or more repetitions.
 * `?` - If `R` is a regular expression, then `R?` matches 1 or no appearances if `R`

 The following character classes are supported as of current
 * `.` - represents any character.
 * `[]` - represents any of the characters inside these brackets (e.g., `[abc]` would mean 'a', 'b' or 'c').
  * `-` - use inside square brackets to denote a range of characters (e.g., `[a-z]` would mean any character from 'a' to 'z').

 `/` is an escape character, it can be used to escape any character that would normally represent an operator or character class* (e.g., `/*` would match "*"), and can also represent the following control codes:
 * `/n` - recognises new line ascii character.
 * `/r` - recognises return carriage ascii character.
 * `/t` - recognises tab ascii character.

 *Note: What requires escaping depends one whether you're inside `[]` or not. For example,
 `*` does not need escaping when inside `[]`, but `-` does.

## Features to be added
 * Substring matching

## Potential future features
 * Subexpressions (probably with `{}`).
 * UTF8 support
 * Any other operators I can think of or remember that are commonly used or are useful.
