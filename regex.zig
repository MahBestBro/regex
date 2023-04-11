//Copyright statement at the bottom of this file

const std = @import("std");

const assert = std.debug.assert;


fn memContains(comptime T: type, mem: []const T, target: T) bool
{
    for (mem) |val| if (val == target) return true;
    return false;
}


const ParseError = error
{
    BinaryOpWithOneOrNoArgs,
    UnaryOpWithNoArg,
    MismatchedClosingParenthesis,
    MismatchedOpeningParenthesis,
    MismatchedClosingSquareBracket,
    MismatchedOpeningSquareBracket,
    UnrecognisedEscape,
    MissingRangeBoundary,
    ReversedRange,
    InvalidHex,
};

const MAX_CHARS = std.math.maxInt(u8) / 2 + 1;

const Symbol = union(enum)
{
    char: u8,
    range: Range,
    wildcard: void,

    pub fn match(self: Symbol, char: u8) bool
    {
        return switch (self)
        {
            .char => |c| c == char,
            .wildcard => true,
            .range => |range| range.char_flags[char] 
        };
    }
};

const Operator = enum
{
    BRACKET, //Perhaps get rid of? Only needed in regexToPostfix
    CONCAT,
    OR,
    KLEENE_STAR,
    OPTIONAL,
    PLUS
};

fn parseControlCode(regex_str: []const u8, at: usize) ParseError!struct{char: u8, offset: usize}
{
    return switch (regex_str[at])
    {
        'n' => .{ .char = '\n', .offset = 0 },
        'r' => .{ .char = '\r', .offset = 0 },
        't' => .{ .char = '\t', .offset = 0 },
        'x' => blk: 
        {  
            const hex = regex_str[at + 1..at + 3];
            const result = std.fmt.parseInt(u8, hex, 16) catch
            {
                return ParseError.InvalidHex;
            };
            break :blk .{ .char = result, .offset = 2 };
        },
        else => return ParseError.UnrecognisedEscape
    };
}

//NOTE: Did some benchmarks on compiling large regular expressions with a lot of ranges,
//and it seems to slow down a lot when mixed with the *, + and ? operators. Also, a 
//"range set" implementation seemed to be approximately 1.5-2x, faster on average, though
//idk whether this is worth swapping as the range set version is more complicated code wise
//and may not matter in practical use.   
const Range = struct
{
    char_flags: [MAX_CHARS]bool,

    pub fn init(str: []const u8) ParseError!Range
    {
        var result: Range = undefined;
        std.mem.set(bool, &result.char_flags, false);
        
        var i: usize = 0; 
        var prev_at: usize = 0;
        while (i < str.len) : (i += 1)
        { 
            switch(str[i])
            {
                '-' => 
                {
                    if (i == 0 or i == str.len - 1) return ParseError.MissingRangeBoundary;

                    //Holy shit this is ugly af hnnnngh
                    const min = blk:
                    {
                        const escaped = str[i - 1];
                        if (i - prev_at > 1 and 
                            escaped != '-' and escaped != '/' and escaped != ']')
                        {
                            const cc = parseControlCode(str, prev_at + 1) catch unreachable;
                            break :blk cc.char;
                        }
                        break :blk str[i - 1];
                    };
                        
                    const max = blk:
                    {
                        if (str[i + 1] != '/') break :blk str[i + 1];

                        const escaped = str[i + 2];
                        if (escaped != '-' and escaped != '/' and escaped != ']')
                        {
                            break :blk (try parseControlCode(str, i + 2)).char;
                        }
                        break :blk str[i + 2];
                    };

                    if (min > max) return ParseError.ReversedRange;

                    //NOTE: The else case should include the start and end of the range
                    std.mem.set(bool, result.char_flags[min + 1..max], true);
                    prev_at = i;
                },
                '/' =>
                {
                    assert(i < str.len - 1);

                    prev_at = i;
                    i += 1;
                    var next_char = str[i];
                    if (next_char != '-' and next_char != '/' and next_char != ']') 
                    {
                        const cc_parse_result = try parseControlCode(str, i);
                        next_char = cc_parse_result.char;
                        i += cc_parse_result.offset;
                    }
                    else
                    {
                        prev_at = i;
                    }

                    result.char_flags[next_char] = true;
                },
                else => 
                {
                    result.char_flags[str[i]] = true;
                    prev_at = i;
                },
            }
        }

        return result;
    }
};


const PostfixElement = union(enum)
{
    symbol: Symbol,
    op: Operator
};

//Converts a regex string to postfix notation.
//TODO: Better error result (maybe have like an "ErrorInfo" struct and return that?)
fn regexToPostfix(
    regex_str: []const u8, 
    postfix_allocator: std.mem.Allocator
) ![]PostfixElement
{
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    var result_at: usize = 0;
    var result = try allocator.alloc(PostfixElement, 2 * regex_str.len - 1);
    var stack = std.ArrayList(Operator).init(allocator);
    var last_added_is_expr = false;
    var i: usize = 0;
    while (i < regex_str.len) : (i += 1)
    {
        const char = regex_str[i];

        const unconcatable = ")|*?+";
        const will_concat = i < regex_str.len - 1 and 
                            !memContains(u8, unconcatable, regex_str[i + 1]); 
        
        switch (char)
        {
            '.' =>
            {
                result[result_at] = PostfixElement{ .symbol = Symbol{ .wildcard = {} } };
                result_at += 1;
                last_added_is_expr = true;
                if (will_concat) try stack.append(.CONCAT);
            },
            '/' => 
            {
                if (i == regex_str.len - 1) return ParseError.UnrecognisedEscape;
                i += 1;

                const special_chars = "()|*?+./[]";
                var next_char = regex_str[i];
                if (!memContains(u8, special_chars, next_char))
                {
                    const cc_parse_result = try parseControlCode(regex_str, i);
                    next_char = cc_parse_result.char;
                    i += cc_parse_result.offset;
                }
                
                result[result_at] = .{ .symbol = Symbol{ .char = next_char } };
                result_at += 1;
                last_added_is_expr = true;
                
                //NOTE: Do not use will_concat here as this relies on the *next* next 
                //character
                if (i < regex_str.len - 1 and 
                    !memContains(u8, unconcatable, regex_str[i + 1])) 
                {
                    try stack.append(.CONCAT);
                }
            },
            '(' => try stack.append(.BRACKET),
            ')' => 
            {
                var found_closing_bracket = false;
                while (stack.popOrNull()) |op|
                {
                    if (op == .BRACKET)
                    {
                        found_closing_bracket = true;
                        break;
                    }

                    result[result_at] = PostfixElement{ .op = op };
                    result_at += 1;
                }
                if (!found_closing_bracket) return ParseError.MismatchedClosingParenthesis;

                last_added_is_expr = true;
                if (will_concat) try stack.append(.CONCAT);
            },
            '[' =>
            {
                if (i == regex_str.len - 1) return ParseError.MismatchedOpeningSquareBracket;

                i += 1;
                const start = i;
                while (i < regex_str.len and regex_str[i] != ']') : (i += 1)
                {
                    //NOTE: This skips over any escaped ']' characters
                    i += @boolToInt(regex_str[i] == '/'); 
                }
                if (i == regex_str.len and (regex_str[i - 1] != ']' or regex_str[i - 2] == '/')) 
                    return ParseError.MismatchedOpeningSquareBracket;
                const end = i;

                result[result_at] = PostfixElement
                { 
                    .symbol = Symbol{ .range = try Range.init(regex_str[start..end]) } 
                };
                result_at += 1;

                last_added_is_expr = true;
                const range_will_concat = i < regex_str.len - 1 and 
                                          !memContains(u8, unconcatable, regex_str[i + 1]);
                if (range_will_concat) try stack.append(.CONCAT);
            },
            ']' => return ParseError.MismatchedClosingSquareBracket,
            '|' =>
            {
                if (i == 0 or i == regex_str.len - 1) return ParseError.BinaryOpWithOneOrNoArgs;

                while (stack.items.len > 0 and 
                       stack.items[stack.items.len - 1] != .BRACKET)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }

                last_added_is_expr = false;
                try stack.append(.OR);
            },
            '*' => 
            {
                if (i == 0 or !last_added_is_expr) return ParseError.UnaryOpWithNoArg;

                result[result_at] = PostfixElement{ .op = .KLEENE_STAR };
                result_at += 1;
                if (stack.items.len > 0 and stack.items[stack.items.len - 1] == .CONCAT)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }

                last_added_is_expr = false;
                if (will_concat) try stack.append(.CONCAT);
            },
            '+' =>
            {
                if (i == 0 or !last_added_is_expr) return ParseError.UnaryOpWithNoArg;

                result[result_at] = PostfixElement{ .op = .PLUS };
                result_at += 1;
                if (stack.items.len > 0 and stack.items[stack.items.len - 1] == .CONCAT)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }

                last_added_is_expr = false;
                if (will_concat) try stack.append(.CONCAT);
            },
            '?' =>
            {
                if (i == 0 or !last_added_is_expr) return ParseError.UnaryOpWithNoArg;

                result[result_at] = PostfixElement{ .op = .OPTIONAL };
                result_at += 1;
                if (stack.items.len > 0 and stack.items[stack.items.len - 1] == .CONCAT)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }

                last_added_is_expr = false;
                if (will_concat) try stack.append(.CONCAT);
            },
            //Maybe add catch case for weird characters such as the ones before ' '???
            else => 
            {
                assert(char != ']');

                result[result_at] = PostfixElement{ .symbol = Symbol{ .char = char } };
                result_at += 1;
                last_added_is_expr = true;
                if (will_concat) try stack.append(.CONCAT);
            },
        }
    }


    while (stack.popOrNull()) |op|
    {
        if (op == .BRACKET) return ParseError.MismatchedOpeningParenthesis;
        result[result_at] = PostfixElement{ .op = op };
        result_at += 1;
    }
    assert(result_at == 1 or result[result_at - 1] != .symbol);

    return try postfix_allocator.dupe(PostfixElement, result[0..result_at]);
}

const NfaState = struct
{
    const NfaTransition = struct
    {
        symbol: ?Symbol, //If null, then it is an empty transition
        next_id: usize,
    };

    id: usize,

    //TODO: Maybe edit double just to be 2 pointers? The symbol is always null. Also
    //maybe edit single just to be a transition with a normal symbol as they're never
    //null.
    transitions: union(enum)
    {
        single: NfaTransition,
        double: [2]NfaTransition,
        final: void
    },

    pub fn single(id: usize, symbol: Symbol, next_id: usize) NfaState
    {
        return NfaState
        {
            .id = id,
            .transitions = 
            .{
                .single = .{ .symbol = symbol, .next_id = next_id },
            }
        };
    }

    pub fn double(id: usize, next1_id: usize, next2_id: usize) NfaState
    {
        return NfaState
        {
            .id = id,
            .transitions = 
            .{
                .double = 
                .{
                    .{ .symbol = null, .next_id = next1_id },
                    .{ .symbol = null, .next_id = next2_id },
                }
            }
        };
    }

    pub fn final(id: usize) NfaState
    {
        return NfaState{ .id = id, .transitions = .{ .final = {} } };
    }
};

//NOTE: If you want to deallocate this, do so by deallocating 'state_pool' as that
//is the only allocated member
const NFA = struct
{
    start: NfaState, 
    state_pool: []NfaState
};

//Creates an NFA given a regex string. 
//
//Does so by converting the string to postfix notation, then goes through the string, 
//using a stack to hold NFA 'fragments' (partially constructed NFAs), and then 
//connecting them up as they get popped off the stack. At the end there should be
//only one fragment on the stack which is the entire NFA.
fn createNFA(str: []const u8, allocator: std.mem.Allocator) !NFA
{    
    const PtrList = std.ArrayList(*usize);

    const Frag = struct
    {
        start_id: usize,
        dangling_ptrs: PtrList
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var arena_allocator = arena.allocator();

    const postfix = try regexToPostfix(str, arena_allocator);

    const state_pool = try arena_allocator.alloc(NfaState, str.len + 1);
    var new_state_index: usize = 0; 

    //TODO: Properly handle regex parse errors (potentially take outside function?)
    var stack = std.ArrayList(Frag).init(arena_allocator);
    for (postfix) |el|
    {
        if (el == .symbol)
        {
            var new_state = &state_pool[new_state_index];
            new_state.* = NfaState.single(new_state_index, el.symbol, undefined);
            new_state_index += 1;
            
            var frag = Frag
            { 
                .start_id = new_state.id, 
                .dangling_ptrs = PtrList.init(arena_allocator) 
            };
            try frag.dangling_ptrs.append(&new_state.transitions.single.next_id);
            try stack.append(frag);

            continue;
        }

        switch (el.op)
        {
            .CONCAT => 
            {
                assert(stack.items.len >= 2);

                const rhs = stack.pop();
                var lhs = &stack.items[stack.items.len - 1]; 
                for (lhs.dangling_ptrs.items) |ptr| ptr.* = rhs.start_id;
                lhs.dangling_ptrs.clearRetainingCapacity();
                try lhs.dangling_ptrs.appendSlice(rhs.dangling_ptrs.items);
            },
            .OR =>
            {
                const lhs = stack.pop(); 
                const rhs = stack.pop();

                var new_state = &state_pool[new_state_index];
                new_state.* = NfaState.double(new_state_index, lhs.start_id, rhs.start_id);
                new_state_index += 1;

                var frag = Frag
                { 
                    .start_id = new_state.id, 
                    .dangling_ptrs = PtrList.init(arena_allocator) 
                };
                try frag.dangling_ptrs.appendSlice(lhs.dangling_ptrs.items);
                try frag.dangling_ptrs.appendSlice(rhs.dangling_ptrs.items);
                try stack.append(frag);
            },
            .KLEENE_STAR =>
            {
                var popped = stack.pop();
                
                var new_state = &state_pool[new_state_index];
                new_state.* = NfaState.double(new_state_index, popped.start_id, undefined);
                new_state_index += 1;
                
                for (popped.dangling_ptrs.items) |ptr| ptr.* = new_state.id;

                var frag = Frag
                { 
                    .start_id = new_state.id, 
                    .dangling_ptrs = PtrList.init(arena_allocator) 
                };
                try frag.dangling_ptrs.append(&new_state.transitions.double[1].next_id);
                try stack.append(frag);
            },
            .OPTIONAL =>
            {
                var popped = stack.pop();
                
                var new_state = &state_pool[new_state_index];
                new_state.* = NfaState.double(new_state_index, popped.start_id, undefined);
                new_state_index += 1;

                var frag = Frag
                { 
                    .start_id = new_state.id, 
                    .dangling_ptrs = PtrList.init(arena_allocator) 
                };
                try frag.dangling_ptrs.appendSlice(popped.dangling_ptrs.items);
                try frag.dangling_ptrs.append(&new_state.transitions.double[1].next_id);
                try stack.append(frag);

            },
            .PLUS =>
            {
                var top = &stack.items[stack.items.len - 1];

                var new_state = &state_pool[new_state_index];
                new_state.* = NfaState.double(new_state_index, top.start_id, undefined);
                new_state_index += 1;

                for (top.dangling_ptrs.items) |ptr| ptr.* = new_state.id;
                top.dangling_ptrs.clearRetainingCapacity();
                
                try top.dangling_ptrs.append(&new_state.transitions.double[1].next_id);
            },
            else => unreachable,
        }
    }

    assert(stack.items.len == 1);

    var result = stack.pop();
    const match_state = &state_pool[new_state_index];
    match_state.* = NfaState.final(new_state_index);
    new_state_index += 1;
    for (result.dangling_ptrs.items) |ptr| ptr.* = match_state.id;

    //std.debug.print("num states for {s}: {d}\n", .{str, new_state_index});

    return NFA
    { 
        .start = state_pool[result.start_id], 
        .state_pool = try allocator.dupe(NfaState, state_pool[0..new_state_index]) 
    };
}


const DFA = struct
{
    table: [][MAX_CHARS]?usize,
    final_state_map: []bool,
};

//Adds all possible states reachable by empty transitions to a state set. This is 
//just a helper function for NFAtoDFA.
//
//'start' is the state to start at when traversing empty transitions.
//'arena_allocator' should be an allocator from std.heap.ArenaAllocator. 
fn fillEmptyTransitions(
    state_set: *std.DynamicBitSet, 
    start: NfaState,
    nfa: NFA, 
    arena_allocator: std.mem.Allocator
) !void
{
    var states_to_traverse = std.ArrayList(NfaState).init(arena_allocator);
    try states_to_traverse.append(start);
    var next_to_traverse: usize = 0;

    //Go through all the empty transitions
    while (next_to_traverse < states_to_traverse.items.len)
    {
        for (states_to_traverse.items[next_to_traverse..]) |state|
        {
            state_set.set(state.id);
            switch(state.transitions)
            {
                .single => |t| assert(t.symbol != null),
                .double => |t|
                {
                    assert(t[0].symbol == null and t[1].symbol == null);

                    try states_to_traverse.append(nfa.state_pool[t[0].next_id]);
                    try states_to_traverse.append(nfa.state_pool[t[1].next_id]);
                },
                .final => {}
            }
            next_to_traverse += 1;
        }
    }
}

//Converts an NFA to a DFA.
//
//Starting at the start state of the NFA, a new row is added to the DFA's state
//table. Then for each symbol in the NFA's alphabet (right now this is literally just 
//the lowercase emglish alphabet), finds all possible states that can be reached 
//via that symbol (including taking empty transitions afterwards) and adds them to a 
//set of states, which is then treated as a state in the DFA. If this state set is 
//new (i.e., a new state has been added to the DFA), a new row is added to the DFA's 
//state table. This keeps happening until no new rows are added to the DFA's state
//table. If a state set has a final state in it, it is considered a final state in
//the DFA.
//
//Note that in the resulting DFA, the state sets are just represented with their
//index into the DFA table as you only need the actual set when constructing the
//DFA.
fn NFAtoDFA(nfa: NFA, allocator: std.mem.Allocator) !DFA
{
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var arena_allocator = arena.allocator();

    var state_sets = std.ArrayList(std.DynamicBitSet).init(arena_allocator);
    var state_table = std.ArrayList([MAX_CHARS]?usize).init(arena_allocator);  
    
    //Add first state of DFA
    try state_sets.append(try std.DynamicBitSet.initEmpty(
        arena_allocator, 
        nfa.state_pool.len
    ));
    try fillEmptyTransitions(&state_sets.items[0], nfa.start, nfa, arena_allocator);
    try state_table.append([_]?usize{null} ** MAX_CHARS);

    var row_to_fill: usize = 0;
    while (row_to_fill < state_table.items.len) : (row_to_fill += 1)
    {
        var char: u8 = 0;
        while (char < MAX_CHARS) : (char += 1)
        {
            var next_state_set = try std.DynamicBitSet.initEmpty(
                arena_allocator, 
                nfa.state_pool.len
            );
            
            //Get all states reachable via transitions with char, starting at all
            //the states in the current state set.
            var current_states = state_sets.items[row_to_fill].iterator(.{});
            while (current_states.next()) |state_id|
            {
                const current_transitions = nfa.state_pool[state_id].transitions;
                switch(current_transitions)
                {
                    .single => |t| if (t.symbol.?.match(char)) next_state_set.set(t.next_id),
                    .double => |t| assert(t[0].symbol == null and t[1].symbol == null),
                    .final => continue
                }
            } 
            
            //Find states reachable from empty transitions
            var it = next_state_set.iterator(.{});
            while (it.next()) |state_id|
            {
                const state = nfa.state_pool[state_id];
                try fillEmptyTransitions(&next_state_set, state, nfa, arena_allocator);
            }
            
            //Get row index of next state set
            var next_state_set_row: ?usize = null;
            for (state_sets.items, 0..) |set, i|
            {
                if (set.eql(next_state_set)) 
                {
                    next_state_set_row = i;
                    break;
                }
            }

            //Write next state set into table
            if (next_state_set_row) |row| 
            {
                state_table.items[row_to_fill][char] = row;
            }
            else
            {
                //Set is new so add a new row to table
                state_table.items[row_to_fill][char] = state_sets.items.len;
                try state_sets.append(next_state_set);
                try state_table.append([_]?usize{null} ** MAX_CHARS);
            } 
        }
    }


    var dfa = DFA
    { 
        .table = try allocator.dupe([MAX_CHARS]?usize, state_table.items),
        .final_state_map = try allocator.alloc(bool, state_table.items.len),
    }; 

    //Fill out the final state map by finding all state sets which have a final state in them
    for (state_sets.items, 0..) |set, set_index|
    {
        var states = set.iterator(.{});
        while (states.next()) |state_id| 
        {
            if (nfa.state_pool[state_id].transitions == .final)
            { 
                dfa.final_state_map[set_index] = true; 
                break;
            }
        }
    }

    return dfa;
}

//TODO: Make this file scope?
pub const Regex = struct
{
    //NOTE: From benchmarks I've done myself at least, using a DFA is faster for matching
    //with the added cost of a little bit more compile time (though the speed up you get
    //looks to be much worth the extra compile time) and memory. If need be, we could
    //potentially have an option to just use the NFA, though I doubt this will ever be
    //a serious need.
    dfa: DFA,  
    allocator: std.mem.Allocator,

    pub fn compile(regex_str: []const u8, allocator: std.mem.Allocator) !Regex
    {
        const nfa = try createNFA(regex_str, allocator);
        defer allocator.free(nfa.state_pool);
        return Regex{ .dfa = try NFAtoDFA(nfa, allocator), .allocator = allocator };
    }

    //Returns true if entirety of 'str' matches the regular expression, else false.
    pub fn match(self: Regex, str: []const u8) bool
    {
        var current_state: usize = 0;
        for (str) |input_char| 
        {
            const transition = self.dfa.table[current_state][input_char];
            if (transition) |next_state| current_state = next_state else return false;
        }
        return self.dfa.final_state_map[current_state];
    }

    //Returns true if any substring of 'str' matches the regular expression, else false.
    pub fn substringMatch(self: Regex, str: []const u8) bool
    {
        for (str, 0..) |front_char, i|
        {
            const first_transition = self.dfa.table[0][front_char];
            if (first_transition != null)
            {
                var current_state: usize = 0;
                for(str[i..]) |input_char|
                {
                    if (self.dfa.final_state_map[current_state]) return true;

                    const transition = self.dfa.table[current_state][input_char];   
                    if (transition) |next_state| current_state = next_state else break;
                }

                if (self.dfa.final_state_map[current_state]) return true;
            }
        }

        return false;
    }

    pub fn deinit(self: Regex) void
    {
        self.allocator.free(self.dfa.table);
        self.allocator.free(self.dfa.final_state_map);
    }
};






const testing = std.testing;

fn pfChar(char: u8) PostfixElement
{
    return PostfixElement{ .symbol = Symbol{ .char = char } };
}

test "regexToPostfix"
{
    //Maybe make a helper function that makes manual PostfixElement arrays easier
    //to type out

    const test1 = try regexToPostfix("a(bb)|a", testing.allocator);
    defer testing.allocator.free(test1);
    try testing.expectEqualSlices(
        PostfixElement, 
        &[_]PostfixElement{ 
            pfChar('a'), pfChar('b'), pfChar('b'), .{ .op = .CONCAT },
            .{ .op = .CONCAT }, pfChar('a'), .{ .op = .OR }
        },
        test1
    );

    const test2 = try regexToPostfix("a(bb)*a", testing.allocator);
    defer testing.allocator.free(test2);
    try testing.expectEqualSlices(
        PostfixElement, 
        &[_]PostfixElement{ 
            pfChar('a'), pfChar('b'), pfChar('b'), .{ .op = .CONCAT },
            .{ .op = .KLEENE_STAR }, .{ .op = .CONCAT }, pfChar('a'), 
            .{ .op = .CONCAT }
        },
        test2
    );

    const test3 = try regexToPostfix("b|ab(a*|b)", testing.allocator);
    defer testing.allocator.free(test3);
    try testing.expectEqualSlices(
        PostfixElement, 
        &[_]PostfixElement{ 
            pfChar('b'), pfChar('a'), pfChar('b'), pfChar('a'), 
            .{ .op = .KLEENE_STAR }, pfChar('b'), .{ .op = .OR }, 
            .{ .op = .CONCAT }, .{ .op = .CONCAT }, .{ .op = .OR }, 
        },
        test3
    );
}

fn nfaHasDuplicateIDs(nfa: NFA) bool
{
    for (nfa.state_pool, 0..) |a, i|
    {
        for (nfa.state_pool[i+1..nfa.state_pool.len]) |b| 
        {
            if (a.id == b.id) return true;
        }
    }
    return false; 
}

test "createNFA"
{
    const single = try createNFA("a", testing.allocator);
    defer testing.allocator.free(single.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(single));

    {
        const first = single.start;
        try testing.expect(first.transitions == .single);
        try testing.expect(first.transitions.single.symbol.?.char == 'a');
        const last = single.state_pool[first.transitions.single.next_id];
        try testing.expect(last.transitions == .final);
    }

    const concat = try createNFA("ab", testing.allocator);
    defer testing.allocator.free(concat.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(concat));

    {
        const first = concat.start;
        try testing.expect(first.transitions == .single);
        try testing.expect(first.transitions.single.symbol.?.char == 'a');
        const second = concat.state_pool[first.transitions.single.next_id];
        try testing.expect(second.transitions == .single);
        try testing.expect(second.transitions.single.symbol.?.char == 'b');
        const last = concat.state_pool[second.transitions.single.next_id];
        try testing.expect(last.transitions == .final);
    }


    const disjunction = try createNFA("a|b", testing.allocator);
    defer testing.allocator.free(disjunction.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(disjunction));

    {
        const first = disjunction.start;
        try testing.expect(first.transitions == .double);
        
        const a_route = disjunction.state_pool[first.transitions.double[1].next_id];
        try testing.expect(a_route.transitions == .single);
        try testing.expect(a_route.transitions.single.symbol.?.char == 'a');
        const b_route = disjunction.state_pool[first.transitions.double[0].next_id];
        try testing.expect(b_route.transitions == .single);
        try testing.expect(b_route.transitions.single.symbol.?.char == 'b');
        try testing.expect(a_route.transitions.single.next_id == b_route.transitions.single.next_id);

        const last = disjunction.state_pool[a_route.transitions.single.next_id];
        try testing.expect(last.transitions == .final);
    }


    const star = try createNFA("a*", testing.allocator);
    defer testing.allocator.free(star.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(star));

    {
        const first = star.start;
        try testing.expect(first.transitions == .double);
        
        const loop = star.state_pool[first.transitions.double[0].next_id];
        try testing.expect(loop.transitions == .single);
        try testing.expect(loop.transitions.single.symbol.?.char == 'a');
        try testing.expectEqual(loop.transitions.single.next_id, first.id);

        const last = star.state_pool[first.transitions.double[1].next_id];
        try testing.expect(last.transitions == .final);
    }

    const optional = try createNFA("a?", testing.allocator);
    defer testing.allocator.free(optional.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(optional));

    {
        const first = optional.start;
        try testing.expect(first.transitions == .double);
        
        const symbol_route = optional.state_pool[first.transitions.double[0].next_id];
        try testing.expect(symbol_route.transitions == .single);
        try testing.expect(symbol_route.transitions.single.symbol.?.char == 'a');

        const last = optional.state_pool[first.transitions.double[1].next_id];
        try testing.expectEqual(symbol_route.transitions.single.next_id, last.id);
        try testing.expect(last.transitions == .final);
    }

    const plus = try createNFA("a+", testing.allocator);
    defer testing.allocator.free(plus.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(plus));

    {
        const first = plus.start;
        try testing.expect(first.transitions == .single);
        try testing.expect(first.transitions.single.symbol.?.char == 'a');
        
        const loop = plus.state_pool[first.transitions.single.next_id];
        try testing.expect(loop.transitions == .double);
        try testing.expectEqual(loop.transitions.double[0].next_id, first.id);

        const last = plus.state_pool[loop.transitions.double[1].next_id];
        try testing.expect(last.transitions == .final);
    }

    const big_boi = try createNFA("b|ab(a*|b)", testing.allocator);
    defer testing.allocator.free(big_boi.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(big_boi));
}

test "sanity check" 
{
    const r = try Regex.compile("a", testing.allocator);
    defer r.deinit();
    try testing.expect(r.match("a"));
    try testing.expect(!r.match("b"));
}

test "concat"
{
    const r = try Regex.compile("abc", testing.allocator);
    defer r.deinit();
    try testing.expect(r.match("abc"));
    try testing.expect(!r.match("a"));
    try testing.expect(!r.match("b"));
    try testing.expect(!r.match("c"));
    try testing.expect(!r.match("ab"));
    try testing.expect(!r.match("bc"));
    try testing.expect(!r.match("ac"));
    try testing.expect(!r.match("aabc"));
}

test "bracket"
{
    const r = try Regex.compile("(a)", testing.allocator);
    defer r.deinit();
    try testing.expect(r.match("a"));
    try testing.expect(!r.match("b"));

    try testing.expectError(ParseError.MismatchedOpeningParenthesis, Regex.compile("abc(", testing.allocator));
    try testing.expectError(ParseError.MismatchedClosingParenthesis, Regex.compile("abc)", testing.allocator));
}

test "or"
{
    const outside_bracket = try Regex.compile("a|(ab)", testing.allocator);
    defer outside_bracket.deinit();
    try testing.expect(outside_bracket.match("a"));
    try testing.expect(outside_bracket.match("ab"));
    try testing.expect(!outside_bracket.match("b"));
    try testing.expect(!outside_bracket.match("aa"));
    
    const inside_bracket = try Regex.compile("a(a|b)", testing.allocator);
    defer inside_bracket.deinit();
    try testing.expect(inside_bracket.match("aa"));
    try testing.expect(inside_bracket.match("ab"));
    try testing.expect(!inside_bracket.match("a"));
    try testing.expect(!inside_bracket.match("b"));

    const multiple_ors = try Regex.compile("a|b|c", testing.allocator);
    defer multiple_ors.deinit();
    try testing.expect(multiple_ors.match("a"));
    try testing.expect(multiple_ors.match("b"));
    try testing.expect(multiple_ors.match("c"));
    try testing.expect(!multiple_ors.match(""));
    try testing.expect(!multiple_ors.match("ab"));
    try testing.expect(!multiple_ors.match("bc"));
    try testing.expect(!multiple_ors.match("ac"));

    const precedence_check = try Regex.compile("a|bc*", testing.allocator);
    defer precedence_check.deinit();
    try testing.expect(precedence_check.match("a"));
    try testing.expect(precedence_check.match("b"));
    try testing.expect(precedence_check.match("bc"));
    try testing.expect(precedence_check.match("bcccccccccc"));
    try testing.expect(!precedence_check.match(""));
    try testing.expect(!precedence_check.match("ac"));
    try testing.expect(!precedence_check.match("acc"));

    try testing.expectError(ParseError.BinaryOpWithOneOrNoArgs, Regex.compile("|", testing.allocator));
    try testing.expectError(ParseError.BinaryOpWithOneOrNoArgs, Regex.compile("a|", testing.allocator));
    try testing.expectError(ParseError.BinaryOpWithOneOrNoArgs, Regex.compile("|a", testing.allocator));
}

test "kleene star"
{
    const sanity_check = try Regex.compile("a*", testing.allocator);
    defer sanity_check.deinit();
    const aaaaaa = "a" ** 100;
    for (0..aaaaaa.len) |i| try testing.expect(sanity_check.match(aaaaaa[0..i]));
    try testing.expect(sanity_check.match(aaaaaa));
    try testing.expect(!sanity_check.match("b"));
    try testing.expect(!sanity_check.match("ab"));
    try testing.expect(!sanity_check.match("aab"));

    const start_concat = try Regex.compile("ab*", testing.allocator);
    defer start_concat.deinit();
    const abbbbb = "a" ++ "b" ** 100;
    for (0..abbbbb.len) |i| try testing.expect(start_concat.match(abbbbb[0..i + 1]));
    try testing.expect(!start_concat.match(""));
    try testing.expect(!start_concat.match("b"));
    try testing.expect(!start_concat.match("aa"));
    try testing.expect(!start_concat.match("aba"));
    try testing.expect(!start_concat.match("aab"));

    const end_concat = try Regex.compile("a*b", testing.allocator);
    defer end_concat.deinit();
    const aaaaab = aaaaaa ++ "b";
    for (0..aaaaab.len) |i| 
        try testing.expect(end_concat.match(aaaaab[aaaaab.len - i - 1..aaaaab.len]));
    try testing.expect(!end_concat.match(""));
    try testing.expect(!end_concat.match("a"));
    try testing.expect(!end_concat.match("aa"));
    try testing.expect(!end_concat.match("ba"));
    try testing.expect(!end_concat.match("abb"));

    const bracketed = try Regex.compile("(ab)*", testing.allocator);
    defer bracketed.deinit();
    const ababab = "ab" ** 100;
    {
        var i: usize = 0;
        while (i <= ababab.len) : (i += 2) 
            try testing.expect(bracketed.match(ababab[0..i]));
    }
    try testing.expect(bracketed.match(ababab));
    try testing.expect(!bracketed.match("a"));
    try testing.expect(!bracketed.match("b"));
    try testing.expect(!bracketed.match("aab"));
    try testing.expect(!bracketed.match("abb"));

    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("*", testing.allocator));
    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("*a", testing.allocator));
    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("a?*", testing.allocator));
}

test "optional"
{
    const sanity_check = try Regex.compile("a?", testing.allocator);
    defer sanity_check.deinit();
    try testing.expect(sanity_check.match(""));
    try testing.expect(sanity_check.match("a"));
    try testing.expect(!sanity_check.match("aa"));
    try testing.expect(!sanity_check.match("b"));

    const before_concat = try Regex.compile("a?b", testing.allocator);
    defer before_concat.deinit();
    try testing.expect(before_concat.match("b"));
    try testing.expect(before_concat.match("ab"));
    try testing.expect(!before_concat.match("aab"));
    try testing.expect(!before_concat.match("a"));
    try testing.expect(!before_concat.match(""));

    const after_concat = try Regex.compile("ab?", testing.allocator);
    defer after_concat.deinit();
    try testing.expect(after_concat.match("a"));
    try testing.expect(after_concat.match("ab"));
    try testing.expect(!after_concat.match("abb"));
    try testing.expect(!after_concat.match("b"));
    try testing.expect(!after_concat.match(""));

    const with_bracket = try Regex.compile("(ab)?", testing.allocator);
    defer with_bracket.deinit();
    try testing.expect(with_bracket.match(""));
    try testing.expect(with_bracket.match("ab"));
    try testing.expect(!with_bracket.match("a"));
    try testing.expect(!with_bracket.match("b"));
    try testing.expect(!with_bracket.match("abab"));

    const in_or = try Regex.compile("a|b?", testing.allocator);
    defer in_or.deinit();
    try testing.expect(in_or.match(""));
    try testing.expect(in_or.match("a"));
    try testing.expect(in_or.match("b"));
    try testing.expect(!in_or.match("ab"));

    const with_kleene_star = try Regex.compile("(ab?)*", testing.allocator);
    defer with_kleene_star.deinit();
    try testing.expect(with_kleene_star.match(""));
    try testing.expect(with_kleene_star.match("a"));
    try testing.expect(with_kleene_star.match("ab"));
    try testing.expect(with_kleene_star.match("aa"));
    try testing.expect(with_kleene_star.match("aab"));
    try testing.expect(with_kleene_star.match("aba"));
    try testing.expect(with_kleene_star.match("abab"));
    try testing.expect(!with_kleene_star.match("b"));
    try testing.expect(!with_kleene_star.match("abb"));

    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("?", testing.allocator));
    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("?a", testing.allocator));
    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("a*?", testing.allocator));
}

test "plus"
{
    const sanity_check = try Regex.compile("a+", testing.allocator);
    defer sanity_check.deinit();
    const aaaaaa = "a" ** 100;
    for (0..aaaaaa.len) |i| try testing.expect(sanity_check.match(aaaaaa[0..i+1]));
    try testing.expect(!sanity_check.match(""));
    try testing.expect(!sanity_check.match("b"));
    try testing.expect(!sanity_check.match("ab"));
    try testing.expect(!sanity_check.match("aab"));

    const start_concat = try Regex.compile("ab+", testing.allocator);
    defer start_concat.deinit();
    const abbbbb = "a" ++ "b" ** 100;
    {
        var i: usize = 1;
        while (i < abbbbb.len) : (i += 1) 
            try testing.expect(start_concat.match(abbbbb[0..i + 1]));
    }
    try testing.expect(!start_concat.match(""));
    try testing.expect(!start_concat.match("a"));
    try testing.expect(!start_concat.match("b"));
    try testing.expect(!start_concat.match("aa"));
    try testing.expect(!start_concat.match("aba"));
    try testing.expect(!start_concat.match("aab"));

    const end_concat = try Regex.compile("a+b", testing.allocator);
    defer end_concat.deinit();
    const aaaaab = aaaaaa ++ "b";
    {
        var i: usize = 1;
        while (i < aaaaab.len) : (i += 1) 
            try testing.expect(end_concat.match(aaaaab[aaaaab.len - i - 1..aaaaab.len]));
    }
    try testing.expect(!end_concat.match(""));
    try testing.expect(!end_concat.match("a"));
    try testing.expect(!end_concat.match("b"));
    try testing.expect(!end_concat.match("aa"));
    try testing.expect(!end_concat.match("ba"));
    try testing.expect(!end_concat.match("abb"));

    const bracketed = try Regex.compile("(ab)+", testing.allocator);
    defer bracketed.deinit();
    const ababab = "ab" ** 100;
    {
        var i: usize = 2;
        while (i <= ababab.len) : (i += 2) 
            try testing.expect(bracketed.match(ababab[0..i]));
    }
    try testing.expect(!bracketed.match(""));
    try testing.expect(!bracketed.match("a"));
    try testing.expect(!bracketed.match("b"));
    try testing.expect(!bracketed.match("aab"));
    try testing.expect(!bracketed.match("abb"));

    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("+", testing.allocator));
    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("+a", testing.allocator));
    try testing.expectError(ParseError.UnaryOpWithNoArg, Regex.compile("a?+", testing.allocator));
}

test "wildcard"
{
    const sanity_check = try Regex.compile(".", testing.allocator);
    defer sanity_check.deinit();
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
            try testing.expect(sanity_check.match(&input));
    }
    try testing.expect(!sanity_check.match(""));
    try testing.expect(!sanity_check.match("aa"));
    try testing.expect(!sanity_check.match("bb"));
    try testing.expect(!sanity_check.match("az"));

    const with_concat = try Regex.compile(".b", testing.allocator);
    defer with_concat.deinit();
    {
        var input = [_]u8 {0, 'b'};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
        {
            try testing.expect(with_concat.match(&input));
            try testing.expect(!with_concat.match(input[0..1]));
        }
    }
    try testing.expect(!with_concat.match(""));
    try testing.expect(!with_concat.match("aa"));
    try testing.expect(!with_concat.match("ba"));

    const with_optional = try Regex.compile(".?", testing.allocator);
    defer with_optional.deinit();
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
            try testing.expect(with_optional.match(&input));
    }
    try testing.expect(with_optional.match(""));
    try testing.expect(!with_optional.match("aa"));
    try testing.expect(!with_optional.match("bb"));
    try testing.expect(!with_optional.match("ab"));

    const with_or = try Regex.compile(".|ab", testing.allocator);
    defer with_or.deinit();
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
            try testing.expect(with_or.match(&input));
    }
    try testing.expect(with_or.match("ab"));
    try testing.expect(!with_or.match(""));
    try testing.expect(!with_or.match("aa"));
    try testing.expect(!with_or.match("bb"));
    try testing.expect(!with_or.match("az"));

    const with_star = try Regex.compile(".*", testing.allocator);
    defer with_star.deinit();
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
            try testing.expect(with_star.match(&input));
    }
    {
        var input: [MAX_CHARS]u8 = undefined;
        for (&input, 0..) |*c, i| c.* = @intCast(u8, i);
        try testing.expect(with_star.match(&input));
        for (&input, 0..) |*c, i| c.* = @intCast(u8, MAX_CHARS - i - 1);
        try testing.expect(with_star.match(&input));
    }
    try testing.expect(with_star.match(""));
    try testing.expect(with_star.match("urmom"));

    const with_plus = try Regex.compile(".+", testing.allocator);
    defer with_plus.deinit();
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
            try testing.expect(with_plus.match(&input));
    }
    {
        var input: [MAX_CHARS]u8 = undefined;
        for (&input, 0..) |*c, i| c.* = @intCast(u8, i);
        try testing.expect(with_plus.match(&input));
        for (&input, 0..) |*c, i| c.* = @intCast(u8, MAX_CHARS - i - 1);
        try testing.expect(with_plus.match(&input));
    }
    try testing.expect(!with_plus.match(""));
}

test "escape"
{
    const special_chars = "()|*?+./[]";

    const all_together = try Regex.compile("/(/)/|/*/?/+/.///[/]", testing.allocator);
    defer all_together.deinit();
    try testing.expect(all_together.match(special_chars));

    const individually = try Regex.compile("/(|/)|/||/*|/?|/+|/.|//|/[|/]", testing.allocator);
    defer individually.deinit();
    for (0..special_chars.len) |i| 
        try testing.expect(individually.match(special_chars[i..i+1]));


    const control_codes = [_]u8{'\n', '\r', '\t', 0};

    const control_codes_together = try Regex.compile("/n/r/t/x00", testing.allocator);
    defer control_codes_together.deinit();
    try testing.expect(control_codes_together.match(&control_codes));

    const control_codes_individually = try Regex.compile("/n|/r|/t|/x00", testing.allocator);
    defer control_codes_individually.deinit();
    for (0..control_codes.len) |i| 
        try testing.expect(control_codes_individually.match(control_codes[i..i+1]));

    var hex: u8 = 0;
    var regex = [_]u8{'/', 'x', 0, 0}; 
    while (hex < MAX_CHARS) : (hex += 1)
    {
        var start: usize = 2;
        if (hex < 16)
        {
            regex[2] = '0';
            start += 1;
        }
        _ = try std.fmt.bufPrint(regex[start..4], "{x}", .{hex});

        const hex_compile = try Regex.compile(&regex, testing.allocator);
        defer hex_compile.deinit(); 
        try testing.expect(!hex_compile.match(""));
        try testing.expect(hex_compile.match(&[_]u8{hex}));
    }

    try testing.expectError(ParseError.UnrecognisedEscape, Regex.compile("/p", testing.allocator));
    try testing.expectError(ParseError.UnrecognisedEscape, Regex.compile("/", testing.allocator));
}

test "range"
{
    const alphabet_lower = "abcdefghijklmnopqrstuvwxyz";
    const alphabet_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const qwerty_lower   = "qwertyuiopasdfghjklzxcvbnm";
    const qwerty_upper   = "QWERTYUIOPASDFGHJKLZXCVBNM";
    const digits = "0123456789";
    
    const sanity_check = try Regex.compile("[abc]", testing.allocator);
    defer sanity_check.deinit();
    try testing.expect(sanity_check.match("a"));
    try testing.expect(sanity_check.match("b"));
    try testing.expect(sanity_check.match("c"));
    try testing.expect(!sanity_check.match(""));
    try testing.expect(!sanity_check.match("ab"));
    try testing.expect(!sanity_check.match("bc"));
    try testing.expect(!sanity_check.match("ac"));
    try testing.expect(!sanity_check.match("ba"));
    try testing.expect(!sanity_check.match("ca"));
    try testing.expect(!sanity_check.match("cb"));

    const empty = try Regex.compile("[]", testing.allocator);
    defer empty.deinit();
    try testing.expect(!empty.match(""));

    const lowercase = try Regex.compile("[a-z]", testing.allocator);
    defer lowercase.deinit();
    try testing.expect(!lowercase.match(""));
    for (0..alphabet_lower.len) |i| try testing.expect(lowercase.match(alphabet_lower[i..i+1]));
    for (0..qwerty_lower.len)   |i| try testing.expect(lowercase.match(qwerty_lower[i..i+1]));
    for (0..alphabet_upper.len) |i|  try testing.expect(!lowercase.match(alphabet_upper[i..i+1]));
    for (0..qwerty_upper.len)   |i| try testing.expect(!lowercase.match(qwerty_upper[i..i+1]));
    for (0..digits.len) |i| try testing.expect(!lowercase.match(digits[i..i+1]));
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
        {
            if (std.ascii.isLower(input[0])) {
                try testing.expect(lowercase.match(&input));
            } else {
                try testing.expect(!lowercase.match(&input));
            }
        }
    }

    const uppercase = try Regex.compile("[A-Z]", testing.allocator);
    defer uppercase.deinit();
    try testing.expect(!uppercase.match(""));
    for (0..alphabet_lower.len) |i| try testing.expect(!uppercase.match(alphabet_lower[i..i+1]));
    for (0..qwerty_lower.len)   |i| try testing.expect(!uppercase.match(qwerty_lower[i..i+1]));
    for (0..alphabet_upper.len) |i| try testing.expect(uppercase.match(alphabet_upper[i..i+1]));
    for (0..qwerty_upper.len)   |i| try testing.expect(uppercase.match(qwerty_upper[i..i+1]));
    for (0..digits.len) |i| try testing.expect(!uppercase.match(digits[i..i+1]));
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
        {
            if (std.ascii.isUpper(input[0])) {
                try testing.expect(uppercase.match(&input));
            } else {
                try testing.expect(!uppercase.match(&input));
            }
        }
    }

    const numeric = try Regex.compile("[0-9]", testing.allocator);
    defer numeric.deinit();
    try testing.expect(!numeric.match(""));
    for (0..alphabet_lower.len) |i| try testing.expect(!numeric.match(alphabet_lower[i..i+1]));
    for (0..qwerty_lower.len)   |i| try testing.expect(!numeric.match(qwerty_lower[i..i+1]));
    for (0..alphabet_upper.len) |i| try testing.expect(!numeric.match(alphabet_upper[i..i+1]));
    for (0..qwerty_upper.len)   |i| try testing.expect(!numeric.match(qwerty_upper[i..i+1]));
    for (0..digits.len) |i| try testing.expect(numeric.match(digits[i..i+1]));
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
        {
            if (std.ascii.isDigit(input[0])) {
                try testing.expect(numeric.match(&input));
            } else {
                try testing.expect(!numeric.match(&input));
            }
        }
    }

    const letter = try Regex.compile("[a-zA-Z]", testing.allocator);
    defer letter.deinit();
    try testing.expect(!letter.match(""));
    for (0..alphabet_lower.len) |i| try testing.expect(letter.match(alphabet_lower[i..i+1]));
    for (0..qwerty_lower.len)   |i| try testing.expect(letter.match(qwerty_lower[i..i+1]));
    for (0..alphabet_upper.len) |i| try testing.expect(letter.match(alphabet_upper[i..i+1]));
    for (0..qwerty_upper.len)   |i| try testing.expect(letter.match(qwerty_upper[i..i+1]));
    for (0..digits.len) |i| try testing.expect(!letter.match(digits[i..i+1]));
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
        {
            if (std.ascii.isAlphabetic(input[0])) {
                try testing.expect(letter.match(&input));
            } else {
                try testing.expect(!letter.match(&input));
            }
        }
    }

    const alphanumeric = try Regex.compile("[a-zA-Z0-9]", testing.allocator);
    defer alphanumeric.deinit();
    try testing.expect(!letter.match(""));
    for (0..alphabet_lower.len) |i| try testing.expect(alphanumeric.match(alphabet_lower[i..i+1]));
    for (0..qwerty_lower.len)   |i| try testing.expect(alphanumeric.match(qwerty_lower[i..i+1]));
    for (0..alphabet_upper.len) |i| try testing.expect(alphanumeric.match(alphabet_upper[i..i+1]));
    for (0..qwerty_upper.len)   |i| try testing.expect(alphanumeric.match(qwerty_upper[i..i+1]));
    for (0..digits.len) |i| try testing.expect(alphanumeric.match(digits[i..i+1]));
    {
        var input = [_]u8{0};
        while (input[0] < MAX_CHARS) : (input[0] += 1)
        {
            if (std.ascii.isAlphanumeric(input[0])) {
                try testing.expect(alphanumeric.match(&input));
            } else {
                try testing.expect(!alphanumeric.match(&input));
            }
        }
    }

    const concat = try Regex.compile("a[a-z]", testing.allocator);
    defer concat.deinit();
    try testing.expect(!concat.match(""));
    {
        var input1 = [_]u8{'a', 'a'};
        var input2 = [_]u8{'b', 'a'};
        for (alphabet_lower) |c|
        {
            input1[1] = c;
            input2[1] = c;
            try testing.expect(concat.match(&input1));
            try testing.expect(!concat.match(&input2));
        }
    }

    const disjunct = try Regex.compile("A|[a-z]", testing.allocator);
    defer disjunct.deinit();
    try testing.expect(!disjunct.match(""));
    try testing.expect(disjunct.match("A"));
    for (0..alphabet_lower.len) |i| try testing.expect(disjunct.match(alphabet_lower[i..i+1]));

    const optional = try Regex.compile("[a-z]?", testing.allocator);
    defer optional.deinit();
    try testing.expect(optional.match(""));
    try testing.expect(!optional.match("A"));
    for (0..alphabet_lower.len) |i| try testing.expect(optional.match(alphabet_lower[i..i+1]));

    const star = try Regex.compile("[a-z]*", testing.allocator);
    defer star.deinit();
    try testing.expect(star.match(""));
    try testing.expect(!star.match("A"));
    for (0..alphabet_lower.len) |i| try testing.expect(star.match(alphabet_lower[i..i+1]));
    try testing.expect(star.match(alphabet_lower));
    try testing.expect(!star.match(alphabet_upper));
    try testing.expect(star.match(qwerty_lower));
    try testing.expect(!star.match(qwerty_upper));

    const plus = try Regex.compile("[a-z]+", testing.allocator);
    defer plus.deinit();
    try testing.expect(!plus.match(""));
    try testing.expect(!plus.match("A"));
    for (0..alphabet_lower.len) |i| try testing.expect(star.match(alphabet_lower[i..i+1]));
    try testing.expect(plus.match(alphabet_lower));
    try testing.expect(!plus.match(alphabet_upper));
    try testing.expect(plus.match(qwerty_lower));
    try testing.expect(!plus.match(qwerty_upper));

    const special_chars = try Regex.compile("[*|?+()[.]", testing.allocator);
    defer special_chars.deinit();
    try testing.expect(special_chars.match("*"));
    try testing.expect(special_chars.match("|"));
    try testing.expect(special_chars.match("?"));
    try testing.expect(special_chars.match("+"));
    try testing.expect(special_chars.match("("));
    try testing.expect(special_chars.match(")"));
    try testing.expect(special_chars.match("["));
    try testing.expect(special_chars.match("."));
    try testing.expect(!special_chars.match(""));
    try testing.expect(!special_chars.match("]"));
    try testing.expect(!special_chars.match("-"));
    try testing.expect(!special_chars.match("/"));


    const escape = try Regex.compile("[/-///]]", testing.allocator);
    defer escape.deinit();
    try testing.expect(escape.match("-"));
    try testing.expect(escape.match("/"));
    try testing.expect(escape.match("]"));
    try testing.expect(!escape.match(""));
    try testing.expect(!escape.match("["));
    try testing.expect(!escape.match("-/"));
    try testing.expect(!escape.match("-]"));
    try testing.expect(!escape.match("/]"));

    const escape_range = try Regex.compile("[//-/]]", testing.allocator);
    defer escape_range.deinit();
    try testing.expect(!escape_range.match("")); 
    var char: u8 = '/';
    while (char <= ']') : (char += 1) try testing.expect(escape_range.match(&[_]u8{char}));

    const control_codes = [_]u8{'\n', '\r', '\t', 0};

    const control_code_range = try Regex.compile("[/n/r/t/x00]", testing.allocator);
    defer control_code_range.deinit();
    for (0..control_codes.len) |i| 
        try testing.expect(control_code_range.match(control_codes[i..i+1]));

    const hex_range = try Regex.compile("[/x00-/x7F]", testing.allocator);
    defer hex_range.deinit();
    try testing.expect(!hex_range.match("")); 
    var hex: u8 = 0;
    while (hex < MAX_CHARS) : (hex += 1) try testing.expect(hex_range.match(&[_]u8{hex})); 


    try testing.expectError(ParseError.UnrecognisedEscape, Regex.compile("[/p]", testing.allocator));
    try testing.expectError(ParseError.MismatchedOpeningSquareBracket, Regex.compile("[", testing.allocator));
    try testing.expectError(ParseError.MismatchedOpeningSquareBracket, Regex.compile("[a", testing.allocator));
    try testing.expectError(ParseError.MismatchedOpeningSquareBracket, Regex.compile("[/]", testing.allocator));
    try testing.expectError(ParseError.MismatchedClosingSquareBracket, Regex.compile("]", testing.allocator));
    try testing.expectError(ParseError.MismatchedClosingSquareBracket, Regex.compile("]a", testing.allocator));
    try testing.expectError(ParseError.MismatchedClosingSquareBracket, Regex.compile("/[]", testing.allocator));
    try testing.expectError(ParseError.ReversedRange, Regex.compile("[z-a]", testing.allocator));

}

test "substring match"
{
    const rx = try Regex.compile("ab", testing.allocator);
    defer rx.deinit();
    try testing.expect(rx.substringMatch("abc"));
    try testing.expect(rx.substringMatch("aaaaaaaaaaaaaaab"));
    try testing.expect(rx.substringMatch("aaaaaaaabaaaaaaa"));
    try testing.expect(rx.substringMatch("abababababababab"));
    try testing.expect(rx.substringMatch("jihpehkthe;jabehtiiiiiiiipor"));
    try testing.expect(!rx.substringMatch(""));
    try testing.expect(!rx.substringMatch("a"));
    try testing.expect(!rx.substringMatch("b"));
    try testing.expect(!rx.substringMatch("aa"));
    try testing.expect(!rx.substringMatch("bb"));
    try testing.expect(!rx.substringMatch("baaaaaaaaaaaaaaa"));
    try testing.expect(!rx.substringMatch("r8tg59h8a6b0j4kiupaoejrwhery09wbhj0kpl,a"));
}

//MIT License
//
//Copyright (c) 2023 MahBestBro
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.