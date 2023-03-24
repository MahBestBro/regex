//Copyright statement at the bottom of this file

const std = @import("std");

const assert = std.debug.assert;


//const MAX_CHARS = std.math.maxInt(u8) / 2;
const MAX_CHARS = 26;

//TODO: Move all of these into Regex struct because probs not gonna be used anywhere 
//else?

const PostfixOp = enum
{
    BRACKET, //Perhaps get rid of? Only needed in regexToPostfix
    CONCAT,
    OR,
    KLEENE_STAR,
    OPTIONAL
};

const PostfixElement = union(enum)
{
    char: u8,
    op: PostfixOp
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
    var result = try postfix_allocator.alloc(PostfixElement, 2 * regex_str.len - 1);
    var stack = std.ArrayList(PostfixOp).init(allocator);
    for (regex_str) |char, i|
    {
        const will_concat = i < regex_str.len - 1 and 
                                (std.ascii.isLower(regex_str[i + 1]) or 
                                 regex_str[i + 1] == '('); 
        
        switch (char)
        {
            'a'...'z' => 
            {
                result[result_at] = PostfixElement{ .char = char };
                result_at += 1;
                if (will_concat) try stack.append(.CONCAT);
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
                if (!found_closing_bracket) return error.MissmatchedClosingBracket;

                if (will_concat) try stack.append(.CONCAT);
            },
            '|' =>
            {
                if (i == 0 or i == regex_str.len - 1) return error.BinOpWithOneArg;

                while (stack.items.len > 0 and 
                       stack.items[stack.items.len - 1] != .BRACKET)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }
                try stack.append(.OR);
            },
            '*' => 
            {
                if (i == 0 or !(std.ascii.isLower(regex_str[i - 1]) or regex_str[i - 1] == ')'))
                    return error.UnaryOpWithNoArg;

                result[result_at] = PostfixElement{ .op = .KLEENE_STAR };
                result_at += 1;
                if (stack.items.len > 0 and stack.items[stack.items.len - 1] == .CONCAT)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }
                if (will_concat) try stack.append(.CONCAT);
            },
            '?' =>
            {
                if (i == 0 or !(std.ascii.isLower(regex_str[i - 1]) or regex_str[i - 1] == ')'))
                    return error.UnaryOpWithNoArg;

                result[result_at] = PostfixElement{ .op = .OPTIONAL };
                result_at += 1;
                if (stack.items.len > 0 and stack.items[stack.items.len - 1] == .CONCAT)
                {
                    result[result_at] = PostfixElement{ .op = stack.pop() };
                    result_at += 1;
                }
                if (will_concat) try stack.append(.CONCAT);
            },
            else => return error.UnknownRegexSymbol
        }
    }


    while (stack.popOrNull()) |op|
    {
        result[result_at] = PostfixElement{ .op = op };
        result_at += 1;
    }
    assert(result_at == 1 or result[result_at - 1] != .char);

    return result[0..result_at];
}

const NfaState = struct
{
    const NfaTransition = struct
    {
        char: ?u8, //If null, then it is an empty transition
        next: *NfaState,
    };

    id: usize,

    transitions: union(enum)
    {
        single: NfaTransition,
        double: [2]NfaTransition,
        final: void
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
    const PtrList = std.ArrayList(**NfaState);

    const Frag = struct
    {
        start: *NfaState,
        dangling_ptrs: PtrList
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var arena_allocator = arena.allocator();

    const state_pool = try allocator.alloc(NfaState, str.len + 1);
    var new_state_index: usize = 0; 

    //TODO: Properly handle regex parse errors (potentially take outside function?)
    const postfix = try regexToPostfix(str, arena_allocator);
    var stack = std.ArrayList(Frag).init(arena_allocator);
    for (postfix) |el|
    {
        if (el == .char)
        {
            var new_state = &state_pool[new_state_index];
            new_state.* = NfaState
            { 
                .id = new_state_index,
                .transitions = 
                .{
                    .single = .{ .char = el.char, .next = undefined } 
                }
            };
            new_state_index += 1;
            
            var frag = Frag
            { 
                .start = new_state, 
                .dangling_ptrs = PtrList.init(arena_allocator) 
            };
            try frag.dangling_ptrs.append(&new_state.transitions.single.next);
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
                for (lhs.dangling_ptrs.items) |ptr| ptr.* = rhs.start;
                lhs.dangling_ptrs.clearRetainingCapacity();
                try lhs.dangling_ptrs.appendSlice(rhs.dangling_ptrs.items);
            },
            .OR =>
            {
                const lhs = stack.pop(); 
                const rhs = stack.pop();

                var new_state = &state_pool[new_state_index];
                new_state.* = NfaState 
                { 
                    .id = new_state_index,
                    .transitions = 
                    .{
                        .double = 
                        .{
                            .{ .char = null, .next = lhs.start },
                            .{ .char = null, .next = rhs.start },
                        }
                    }
                };
                new_state_index += 1;

                var frag = Frag
                { 
                    .start = new_state, 
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
                new_state.* = NfaState 
                { 
                    .id = new_state_index,
                    .transitions = 
                    .{
                        .double = 
                        .{
                            .{ .char = null, .next = popped.start },
                            .{ .char = null, .next = undefined },
                        }
                    }
                };
                new_state_index += 1;
                
                for (popped.dangling_ptrs.items) |ptr| ptr.* = new_state;

                var frag = Frag
                { 
                    .start = new_state, 
                    .dangling_ptrs = PtrList.init(arena_allocator) 
                };
                try frag.dangling_ptrs.append(&new_state.transitions.double[1].next);
                try stack.append(frag);
            },
            .OPTIONAL =>
            {
                var popped = stack.pop();
                
                var new_state = &state_pool[new_state_index];
                new_state.* = NfaState 
                { 
                    .id = new_state_index,
                    .transitions = 
                    .{
                        .double = 
                        .{
                            .{ .char = null, .next = popped.start },
                            .{ .char = null, .next = undefined },
                        }
                    }
                };
                new_state_index += 1;

                var frag = Frag
                { 
                    .start = new_state, 
                    .dangling_ptrs = PtrList.init(arena_allocator) 
                };
                try frag.dangling_ptrs.appendSlice(popped.dangling_ptrs.items);
                try frag.dangling_ptrs.append(&new_state.transitions.double[1].next);
                try stack.append(frag);

            },
            else => unreachable,
        }
    }

    assert(stack.items.len == 1);

    var result = stack.pop();
    const match_state = &state_pool[new_state_index];
    match_state.* = NfaState{ .id = new_state_index, .transitions = .{ .final = {} } };
    new_state_index += 1;
    for (result.dangling_ptrs.items) |ptr| ptr.* = match_state;

    return NFA{ .start = result.start.*, .state_pool = state_pool[0..new_state_index] };
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
    arena_allocator: std.mem.Allocator
) !void
{
    var states_to_traverse = std.AutoHashMap(NfaState, void).init(arena_allocator);
    try states_to_traverse.put(start, {});

    //Go through all the empty transitions
    while (states_to_traverse.count() > 0)
    {
        var current_states = states_to_traverse.keyIterator();
        while (current_states.next()) |state|
        {
            state_set.set(state.id);
            switch(state.transitions)
            {
                .single => 
                {
                    const t = state.transitions.single; 
                    if (t.char == null) try states_to_traverse.put(t.next.*, {});

                },
                .double =>
                {
                    const t = state.transitions.double;
                    if (t[0].char == null) try states_to_traverse.put(t[0].next.*, {});
                    if (t[1].char == null) try states_to_traverse.put(t[1].next.*, {});
                },
                .final => {}
            }
            states_to_traverse.removeByPtr(state);
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
    try fillEmptyTransitions(&state_sets.items[0], nfa.start, arena_allocator);
    try state_table.append([_]?usize{null} ** MAX_CHARS);

    var row_to_fill: usize = 0;
    while (row_to_fill < state_table.items.len) : (row_to_fill += 1)
    {
        var char: u8 = 'a';
        while (char < 'a' + MAX_CHARS) : (char += 1)
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
                    .single => 
                    {
                        const t = current_transitions.single; 
                        if (t.char == char) next_state_set.set(t.next.id);
                    },
                    .double =>
                    {
                        const t = current_transitions.double;
                        if (t[0].char == char) next_state_set.set(t[0].next.id);
                        if (t[1].char == char) next_state_set.set(t[1].next.id);
                    },
                    .final => continue
                }
            } 
            
            //Find states reachable from empty transitions
            var it = next_state_set.iterator(.{});
            while (it.next()) |state_id|
            {
                const state = nfa.state_pool[state_id];
                try fillEmptyTransitions(&next_state_set, state, arena_allocator);
            }
            
            //Get row index of next state set
            var next_state_set_row: ?usize = null;
            for (state_sets.items) |set, i|
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
                state_table.items[row_to_fill][char - 'a'] = row;
            }
            else
            {
                //Set is new so add a new row to table
                state_table.items[row_to_fill][char - 'a'] = state_sets.items.len;
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
    for (state_sets.items) |set, set_index|
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

pub const Regex = struct
{
    dfa: DFA,
    allocator: std.mem.Allocator,

    pub fn compile(regex_str: []const u8, allocator: std.mem.Allocator) !Regex
    {
        const nfa = try createNFA(regex_str, allocator);
        defer allocator.free(nfa.state_pool);
        return Regex{ .dfa = try NFAtoDFA(nfa, allocator), .allocator = allocator };
    }

    pub fn match(self: Regex, str: []const u8) bool
    {
        var current_state: usize = 0;
        for (str) |input_char| 
        {
            const transition = self.dfa.table[current_state][input_char - 'a'];
            if (transition) |next_state| current_state = next_state else return false;
        }
        return self.dfa.final_state_map[current_state];
    }

    pub fn deinit(self: Regex) void
    {
        self.allocator.free(self.dfa.table);
        self.allocator.free(self.dfa.final_state_map);
    }
};


//TODO: Benchmark NFA vs DFA matching 

const testing = std.testing;

test "regexToPostfix"
{
    //Maybe make a helper function that makes manual PostfixElement arrays easier
    //to type out

    const test1 = try regexToPostfix("a(bb)|a", testing.allocator);
    defer testing.allocator.free(test1);
    try testing.expectEqualSlices(
        PostfixElement, 
        &[_]PostfixElement{ 
            .{ .char = 'a' }, .{ .char = 'b' }, .{ .char = 'b' }, .{ .op = .CONCAT },
            .{ .op = .CONCAT }, .{ .char = 'a' }, .{ .op = .OR }
        },
        test1
    );

    const test2 = try regexToPostfix("a(bb)*a", testing.allocator);
    defer testing.allocator.free(test2);
    try testing.expectEqualSlices(
        PostfixElement, 
        &[_]PostfixElement{ 
            .{ .char = 'a' }, .{ .char = 'b' }, .{ .char = 'b' }, .{ .op = .CONCAT },
            .{ .op = .KLEENE_STAR }, .{ .op = .CONCAT }, .{ .char = 'a' }, 
            .{ .op = .CONCAT }
        },
        test2
    );

    const test3 = try regexToPostfix("b|ab(a*|b)", testing.allocator);
    defer testing.allocator.free(test3);
    try testing.expectEqualSlices(
        PostfixElement, 
        &[_]PostfixElement{ 
            .{ .char = 'b' }, .{ .char = 'a' }, .{ .char = 'b' }, .{ .char = 'a' }, 
            .{ .op = .KLEENE_STAR }, .{ .char = 'b' }, .{ .op = .OR }, 
            .{ .op = .CONCAT }, .{ .op = .CONCAT }, .{ .op = .OR }, 
        },
        test3
    );
}

fn nfaHasDuplicateIDs(nfa: NFA) bool
{
    for (nfa.state_pool) |a, i|
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
        try testing.expect(first.transitions.single.char.? == 'a');
        const last = first.transitions.single.next.*;
        try testing.expect(last.transitions == .final);
    }

    const concat = try createNFA("ab", testing.allocator);
    defer testing.allocator.free(concat.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(concat));

    {
        const first = concat.start;
        try testing.expect(first.transitions == .single);
        try testing.expect(first.transitions.single.char.? == 'a');
        const second = first.transitions.single.next.*;
        try testing.expect(second.transitions == .single);
        try testing.expect(second.transitions.single.char.? == 'b');
        const last = second.transitions.single.next.*;
        try testing.expect(last.transitions == .final);
    }


    const disjunction = try createNFA("a|b", testing.allocator);
    defer testing.allocator.free(disjunction.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(disjunction));

    {
        const first = disjunction.start;
        try testing.expect(first.transitions == .double);
        
        const a_route = first.transitions.double[1].next.*;
        try testing.expect(a_route.transitions == .single);
        try testing.expect(a_route.transitions.single.char.? == 'a');
        const b_route = first.transitions.double[0].next.*;
        try testing.expect(b_route.transitions == .single);
        try testing.expect(b_route.transitions.single.char.? == 'b');
        try testing.expect(a_route.transitions.single.next == b_route.transitions.single.next);

        const last = a_route.transitions.single.next.*;
        try testing.expect(last.transitions == .final);
    }


    const star = try createNFA("a*", testing.allocator);
    defer testing.allocator.free(star.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(star));

    {
        const first = star.start;
        try testing.expect(first.transitions == .double);
        
        const loop = first.transitions.double[0].next.*;
        try testing.expect(loop.transitions == .single);
        try testing.expect(loop.transitions.single.char.? == 'a');
        try testing.expectEqual(loop.transitions.single.next.*, first);

        const last = first.transitions.double[1].next.*;
        try testing.expect(last.transitions == .final);
    }

    const optional = try createNFA("a?", testing.allocator);
    defer testing.allocator.free(optional.state_pool);
    try testing.expect(!nfaHasDuplicateIDs(optional));

    {
        const first = optional.start;
        try testing.expect(first.transitions == .double);
        
        const char_route = first.transitions.double[0].next.*;
        try testing.expect(char_route.transitions == .single);
        try testing.expect(char_route.transitions.single.char.? == 'a');

        const last = first.transitions.double[1].next.*;
        try testing.expectEqual(char_route.transitions.single.next.*, last);
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
}

test "kleene star"
{
    const sanity_check = try Regex.compile("a*", testing.allocator);
    defer sanity_check.deinit();
    const aaaaaa = "a" ** 100;
    for (aaaaaa) |_, i| try testing.expect(sanity_check.match(aaaaaa[0..i]));
    try testing.expect(sanity_check.match(aaaaaa));
    try testing.expect(!sanity_check.match("b"));
    try testing.expect(!sanity_check.match("ab"));
    try testing.expect(!sanity_check.match("aab"));

    const start_concat = try Regex.compile("ab*", testing.allocator);
    defer start_concat.deinit();
    const abbbbb = "a" ++ "b" ** 100;
    for (abbbbb) |_, i| try testing.expect(start_concat.match(abbbbb[0..i + 1]));
    try testing.expect(!start_concat.match(""));
    try testing.expect(!start_concat.match("b"));
    try testing.expect(!start_concat.match("aa"));
    try testing.expect(!start_concat.match("aba"));
    try testing.expect(!start_concat.match("aab"));

    const end_concat = try Regex.compile("a*b", testing.allocator);
    defer end_concat.deinit();
    const aaaaab = aaaaaa ++ "b";
    for (aaaaab) |_, i| 
        try testing.expect(end_concat.match(aaaaab[aaaaab.len - i - 1..aaaaab.len]));
    try testing.expect(!end_concat.match(""));
    try testing.expect(!end_concat.match("aa"));
    try testing.expect(!end_concat.match("ba"));
    try testing.expect(!end_concat.match("abb"));

    const bracketed = try Regex.compile("(ab)*", testing.allocator);
    defer bracketed.deinit();
    const ababab = "ab" ** 100;
    {
        var i: usize = 0;
        while (i <= ababab.len) : (i += 2) try testing.expect(bracketed.match(ababab[0..i]));
    }
    try testing.expect(bracketed.match(ababab));
    try testing.expect(!bracketed.match("a"));
    try testing.expect(!bracketed.match("b"));
    try testing.expect(!bracketed.match("aab"));
    try testing.expect(!bracketed.match("abb"));
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