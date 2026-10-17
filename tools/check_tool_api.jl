# Static API check for tools/*.jl: does every free global a tool uses resolve against the current Pan API?
#   ./julia_gdal.sh --project=. tools/check_tool_api.jl tools/foo.jl [tools/bar.jl ...]
# For each file: evaluate its using/import lines in a scratch module (so macros + exported names exist), lower every top-level
# expression, collect GlobalRefs of that module, subtract names the file itself defines (assignment / function / struct / for /
# let / const / macro anywhere in the file) and report what is neither defined in the file nor resolvable in the scratch module.
# `Pan.X` / `Pan.Sub.X` qualified references are checked with isdefined as well. Prints "OK" or "UNRESOLVED: a, b".
using Pan

const SKIP_USING = Set([:JLD, :PythonPlot, :Plots])   # packages absent from the project env: their names are reported separately

function defined_names!(s::Set{Symbol}, ex)
    if ex isa Symbol
        return
    elseif ex isa Expr
        h = ex.head
        if h in (:(=), :(+=), :(-=), :(*=), :(/=), :(.=), :function) && !isempty(ex.args)
            lhs_syms!(s, ex.args[1])
        elseif h in (:struct, :macro, :abstract, :primitive)
            lhs_syms!(s, ex.args[h == :struct ? 2 : 1])
        elseif h == :for || h == :generator
            specs = h == :for ? ex.args[1:1] : ex.args[2:end]
            for a in specs
                a isa Expr && a.head == :(=) && lhs_syms!(s, a.args[1])
                a isa Expr && a.head == :block && foreach(b -> b isa Expr && b.head == :(=) && lhs_syms!(s, b.args[1]), a.args)
            end
        elseif h in (:global, :local, :const, :let)
            for a in ex.args; a isa Expr && a.head == :(=) ? lhs_syms!(s, a.args[1]) : (a isa Symbol && push!(s, a)); end
        elseif h in (:import, :using)
            for a in ex.args
                a isa Expr && (a.head == :(:) ? foreach(b -> push!(s, b.args[end]), a.args[2:end]) : (a.head == :as ? push!(s, a.args[2]) : push!(s, a.args[end])))
            end
        elseif h == :->
            lhs_syms!(s, ex.args[1])
        end
        foreach(a -> defined_names!(s, a), ex.args)
    end
end
function lhs_syms!(s::Set{Symbol}, l)
    if l isa Symbol; push!(s, l)
    elseif l isa Expr
        if l.head == :call; lhs_syms!(s, l.args[1]); for a in l.args[2:end]; lhs_syms!(s, a isa Expr && a.head in (:kw, :(::), :parameters) ? (a.head == :parameters ? Expr(:tuple, a.args...) : a.args[1]) : a); end
        elseif l.head == :where; lhs_syms!(s, l.args[1]); for a in l.args[2:end]; lhs_syms!(s, a isa Expr && a.head == :(<:) ? a.args[1] : a); end
        elseif l.head == :(::); lhs_syms!(s, l.args[1])
        elseif l.head == :(.); nothing
        else foreach(a -> lhs_syms!(s, a), l.args); end
    end
end

function globalrefs!(out::Set{Symbol}, mod::Module, x)
    if x isa GlobalRef; x.mod === mod && push!(out, x.name)
    elseif x isa Core.CodeInfo; foreach(a -> globalrefs!(out, mod, a), x.code)
    elseif x isa Expr; foreach(a -> globalrefs!(out, mod, a), x.args)
    elseif x isa Core.ReturnNode; isdefined(x, :val) && globalrefs!(out, mod, x.val)
    elseif x isa Core.GotoIfNot; globalrefs!(out, mod, x.cond)
    end
end

# aliases: `const P = Pan`, `D = Pan.Data`, `import Pan as P` -> alias => ["Data"] path below Pan
function aliases(top)
    al = Dict{Symbol,Vector{Symbol}}(:Pan => Symbol[])
    function chain(e)
        c = Symbol[]
        while e isa Expr && e.head == :(.) && length(e.args) == 2 && e.args[2] isa QuoteNode
            pushfirst!(c, e.args[2].value); e = e.args[1]
        end
        e isa Symbol ? (e, c) : (nothing, c)
    end
    function walk(ex)
        ex isa Expr || return
        if ex.head == :const && ex.args[1] isa Expr; walk(ex.args[1]); return; end
        if ex.head == :(=) && ex.args[1] isa Symbol
            root, c = chain(ex.args[2]); root = ex.args[2] isa Symbol ? ex.args[2] : root
            if root !== nothing && haskey(al, root); al[ex.args[1]] = vcat(al[root], ex.args[2] isa Symbol ? Symbol[] : c); end
        elseif ex.head == :import
            for a in ex.args
                if a isa Expr && a.head == :as && a.args[1] isa Expr && a.args[1].args[1] === :Pan
                    al[a.args[2]] = Symbol.(a.args[1].args[2:end])
                end
            end
        end
        foreach(walk, ex.args)
    end
    walk(top); al
end

function qualified!(out::Set{String}, ex, al)
    ex isa Expr || return
    if ex.head == :(.) && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        chain = Symbol[]; e = ex
        while e isa Expr && e.head == :(.) && length(e.args) == 2 && e.args[2] isa QuoteNode
            pushfirst!(chain, e.args[2].value); e = e.args[1]
        end
        e isa Symbol && haskey(al, e) && push!(out, join(vcat(al[e], chain), "."))
    end
    foreach(a -> qualified!(out, a, al), ex.args)
end

function check(path)
    src = read(path, String)
    top = Meta.parseall(src; filename = path)
    any(a -> a isa Expr && a.head in (:error, :incomplete), top.args) && return "PARSE ERROR"
    # splice in tools/*.jl files this one include()s (so their definitions and using-lines count)
    for mt in eachmatch(r"include\((?:joinpath\(@__DIR__, )?\"([\w./-]+\.jl)\"", src)
        f = joinpath(dirname(path), mt.captures[1]); isfile(f) || continue
        append!(top.args, Meta.parseall(read(f, String); filename = f).args)
    end
    m = Module(:ToolCheck)
    Core.eval(m, :(using Pan))
    missingpk = String[]
    function visit_using(ex)
        ex isa Expr || return
        if ex.head in (:using, :import)
            for a in ex.args
                a = a.head == :(:) ? a.args[1] : a
                a.args[1] in SKIP_USING && (push!(missingpk, string(a.args[1])); continue)
                try Core.eval(m, Expr(ex.head, ex.args...)); catch e; push!(missingpk, sprint(showerror, e)[1:min(end,80)]); end
                break
            end
        else foreach(visit_using, ex.args); end
    end
    foreach(visit_using, top.args)
    defs = Set{Symbol}(); defined_names!(defs, top)
    refs = Set{Symbol}()
    for ex in top.args
        ex isa LineNumberNode && continue
        (ex isa Expr && ex.head in (:using, :import)) && continue
        low = try Meta.lower(m, ex) catch e; (push!(missingpk, "lower: " * sprint(showerror, e)[1:min(end,100)]); continue) end
        low isa Expr && low.head == :error && (push!(missingpk, "lower-error"); continue)
        low isa Expr && low.head == :thunk && globalrefs!(refs, m, low.args[1])
        # nested method bodies
        function walk(x)
            if x isa Core.CodeInfo; for a in x.code; walk(a); end
            elseif x isa Expr; for a in x.args; walk(a); end
            elseif x isa Core.CodeInfo; nothing
            end
            x isa Core.CodeInfo && globalrefs!(refs, m, x)
        end
        low isa Expr && low.head == :thunk && walk(low.args[1])
    end
    unres = sort!([string(s) for s in refs if !(s in defs) && !startswith(string(s), "#") && !(s in (:include, :eval, :ans)) && !Base.invokelatest(isdefined, m, s)])
    q = Set{String}(); qualified!(q, top, aliases(top))
    badq = String[]
    for c in sort!(collect(q))
        o = Pan; ok = true
        for p in split(c, ".")
            if o isa Module && Base.invokelatest(isdefined, o, Symbol(p)); o = getfield(o, Symbol(p)); else ok = false; break; end
        end
        ok || push!(badq, "Pan." * c)
    end
    return (unres, badq, missingpk)
end

for p in ARGS
    r = check(p)
    if r isa String; println(p, ": ", r); continue; end
    unres, badq, mp = r
    println(p, ": ", isempty(unres) && isempty(badq) ? "OK" : "UNRESOLVED: " * join(vcat(unres, badq), ", "),
            isempty(mp) ? "" : "   [not loadable here: " * join(mp, "; ") * "]")
end
