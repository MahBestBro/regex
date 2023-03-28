# zig-regex
 A single file regex library written in and for Zig.

 **Note:** This library is still mostly unfinished, and as a result it cannot do a lot in its current state. For now I would reccomend using another library if you are looking for something robust.  
 
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
 * `*` - If `R` is a regular expression, then `R*` matches 0 or more repitions of `R`.
 * `+` - Same as `*` but 1 or more repetitions.
 * `?` - If `R` is a regular expression, then `R?` matches 1 or no appearances if `R`

 Currently, the following are unique characters with their own special properties:
 * `.` - represents any character. 

## Features to be added
 * Support for all ascii characters (this also implies the addition of `\` as an escape character).
 * `[]` (everything in the square brackets are or'd not concatenated).
    - Also support ranges with `-` (e.g., `[a-z]` would mean anything from `a` to `z`).

## Potential future features
 * Subexpressions (probably with `{}`).
 * Unicode support
 * Any other operators I can think of or remember that are commonly used or are useful.
