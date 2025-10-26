#!/usr/bin/env python3

"""
c2p

A small C-to-Python transpiler for illustrating the basics of compilers to a python programmer. 
"""

import sys, re, textwrap


# Lexer

Token = tuple  # (type, value, line, col)

# regex patterns for creating tokens 
SPEC = [
    #(kind (str), regex pattern)
    ("NUMBER",   r"\d+"),
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP",       r"==|!=|<=|>=|&&|\|\||<<|>>|\+|-|\*|/|%|=|<|>|!"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("SEMI",     r";"),
    ("COMMA",    r","),
    ("WS",       r"[ \t\r]+"),
    ("NEWLINE",  r"\n"),
    ("MISMATCH", r"."),
]

MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name,pat in SPEC))
# kind for KEYWORD[str] is KEYWORD[str].upper()
KEYWORDS = {"int", "return", "if", "else", "while","for"}


def lex(src):
    """
    Produces a token generator.

    Args:
        src (str): The source code user wishes to transpile.

    Yields:
        tuple: The desired token created from the current match. Has the form: (token kind (str), val (str), line (int), column (int))

    Example:
        >>> src = 'int main() {int a = 5; return a;}'
        >>> tokens = tinyc.lex(src)
        >>> next(tokens)
        ('INT', 'int', 1, 1)
        >>> next(tokens)
        ('ID', 'main', 1, 5)
        
    """
    line = 1
    col = 1
    for m in MASTER.finditer(src):
        kind = m.lastgroup #see re.compile methods
        val = m.group()
        if kind == "NEWLINE":
            line += 1
            col = 1
            continue
        if kind == "WS":
            col += len(val)
            continue
        if kind == "ID" and val in KEYWORDS:
            kind = val.upper()
        if kind == "MISMATCH":
            raise SyntaxError(f"Unexpected {val!r} at {line}:{col}")
        yield (kind, val, line, col)
        col += len(val)
    yield ("EOF","",line,col)



# AST nodes

"""
Class definitions for Abstract Syntax Trees(AST).
"""
class Node: pass
class Program(Node):
    def __init__(self, funcs): self.funcs = funcs
class FuncDecl(Node):
    def __init__(self, name, params, body): self.name=name; self.params=params; self.body=body
class Block(Node):
    def __init__(self, stmts): self.stmts=stmts
class VarDecl(Node):
    def __init__(self, name, init): self.name=name; self.init=init
class Return(Node):
    def __init__(self, expr): self.expr=expr
class If(Node):
    def __init__(self, cond, then, els): self.cond=cond; self.then=then; self.els=els
class While(Node):
    def __init__(self, cond, body): self.cond=cond; self.body=body
class ExprStmt(Node):
    def __init__(self, expr): self.expr=expr
# expressions
class Number(Node):
    def __init__(self, val): self.val=int(val)
class Var(Node):
    def __init__(self, name): self.name=name
class Call(Node):
    def __init__(self, name, args): self.name=name; self.args=args
class Unary(Node):
    def __init__(self, op, expr): self.op=op; self.expr=expr
class Binary(Node):
    def __init__(self, op, left, right): self.op=op; self.left=left; self.right=right
class For(Node):
    def __init__(self, index, cond, update, body): self.index = index; self.cond = cond; self.update = update; self.body = body


def viewAST(node, indent=0): 
    """
    Prints a formatted AST.

    Args:
        Node (instance of Node object), indent (int).

    Example:
        >>> src = "int main() {int a = 5; return a;}"
        >>> tokens = lex(src)
        >>> parser = Parser(tokens)
        >>> prog = parser.parse()
        >>> viewAST(node)
        Program
        funcs: [list of 1]
            FuncDecl
                body: 
                Block
                    stmts: [list of 2]
                        VarDecl
                            init: 
                            Number
                                val: 5
                            name: a
                        Return
                            expr: 
                            Var
                                name: a
                name: main
            params: [list of 0]

    Goal:
        A more readable and simple AST viewing option that wat is offered by LLVM.
        
    """

    prefix = "  " * indent
    print(f"{prefix}{type(node).__name__}")
    
    for attr_name in dir(node):
        if attr_name.startswith('__'):
            continue
        attr = getattr(node, attr_name)
        if attr is None:
            continue
        print(f"{prefix}  {attr_name}: ", end="")
        if isinstance(attr, list):
            print(f"[list of {len(attr)}]")
            for att in attr:
                if isinstance(att, Node):
                    viewAST(att, indent + 2)
                else:
                    print(f" {'`' + ('  '* (indent + 2))}{att}")
        elif isinstance(attr, Node):
            print()
            viewAST(attr, indent + 1)
        else:
            print(attr)
        

# Parser

class Parser:
    """
    A recursive descent parser.

    Takes a sequence of tokens(generator produced by lex()) and
    returns an abstract syntax tree.
    """


    def __init__(self, tokens):
        self.tokens = list(tokens) #input tape
        self.pos = 0 #state variable
    
    def cur(self):
        """
        Returns current token.
        """

        return self.tokens[self.pos]
    
    def eat(self, typ=None):
        """
        Assures that the current node is not of a certain type and iterates position if assured.

        Args:
            typ (type)

        Returns:
            current token (tuple)
        """

        t = self.cur()
        if typ and t[0] != typ:
            raise SyntaxError(f"Expected {typ} got {t} at {t[2]}:{t[3]}")
        self.pos += 1
        return t
    
    def peek(self, n=1):
        """
        View the next token in your parser. If the the position of your parser + n
        is greater than the length of the token tape in your parser, it will let you
        know(return token tuple with first index "EOF".

        Args:
            n (int)

        Returns:
            next token (tuple)
        """

        if self.pos+n < len(self.tokens):
            return self.tokens[self.pos+n]
        return ("EOF","",0,0)

    def parse(self):
        """
        Creates a list of FuncDecl objects and returns a Program.
        """

        funcs=[]
        while self.cur()[0] != "EOF":
            funcs.append(self.parse_func())
        return Program(funcs)

    def parse_func(self):
        """
        Parses a function with an integer return type.

        Returns:
            parsed function (tinyc.FuncDecl)

        Example:
            Output of printing the return value of eat in each call
            to Parser.eat when running Parser.parse_func on a Parser(lex(src))
            for src = "int main() {int a = 5; return a;}"

            >>> Parser.parse_func()
            ('INT', 'int', 1, 1)
            main # value of name
            ('LPAREN', '(', 1, 9)
            ('RPAREN', ')', 1, 10)
            #would parse the block but here but eat returns vals arent shown for this parse_block
            <tinyc.FuncDecl object at 0x104ebcbc0>
        """

        self.eat("INT") #We can only compile functions with return type not int.
        name = self.eat("ID")[1]
        self.eat("LPAREN")
        params=[]
        if self.cur()[0] != "RPAREN":
            while True:
                self.eat("INT"); p = self.eat("ID")[1]; params.append(p)
                if self.cur()[0]=="COMMA": self.eat("COMMA"); continue
                break
        self.eat("RPAREN")
        body = self.parse_block()
        return FuncDecl(name, params, body)

    def parse_block(self):
        """
        Parses a block inside of a function and returns a Parser.Block 
        object containing the statements in the block. Gets called upon 
        reaching a token of kind "LBRACE" in the tape (self.cur()[0] == "LBRACE").

        Examples:
            See usage in parse_func and parse_stmt.
        """

        self.eat("LBRACE") #assumes this is the current token.
        stmts=[]
        while self.cur()[0]!="RBRACE":
            stmts.append(self.parse_stmt())
        self.eat("RBRACE")
        return Block(stmts)

    def parse_stmt(self):
        """
        Recursive statement parsing that is usually called after a keyword token or by
        parse_block("LBRACE"). A statement can be thought of as the argument given to a
        keyword. After parsing a keyword token parse_stmt will return a Node subclass associated
        with that keyword. If the current token is not a keyword or "LBRACE" token, returns
        an epression node with the parsed expression found in the statment.

        Returns:
            Node subclass correspoding to current token.

        Called by: 
            parse_block, parse_stmt
        """

        t=self.cur()
        if t[0]=="INT":
            self.eat("INT"); name=self.eat("ID")[1]
            init=None
            if self.cur()[0]=="OP" and self.cur()[1]=="=":
                self.eat("OP"); init=self.parse_expr()
            self.eat("SEMI"); return VarDecl(name, init)
        if t[0]=="RETURN":
            self.eat("RETURN")
            expr = None
            if self.cur()[0]!="SEMI": expr=self.parse_expr()
            self.eat("SEMI"); return Return(expr)
        if t[0]=="IF":
            self.eat("IF"); self.eat("LPAREN"); cond=self.parse_expr(); self.eat("RPAREN")
            then = self.parse_stmt()
            els=None
            if self.cur()[0]=="ELSE":
                self.eat("ELSE"); els=self.parse_stmt()
            return If(cond, then, els)
        if t[0]=="WHILE":
            self.eat("WHILE"); self.eat("LPAREN"); cond=self.parse_expr(); self.eat("RPAREN")
            body = self.parse_stmt(); return While(cond, body)
        if t[0]=="FOR":
            self.eat("FOR"); self.eat("LPAREN")
            index = self.parse_stmt()
            cond = self.parse_expr(); self.eat("SEMI")
            update = self.parse_expr()
            self.eat("RPAREN")
            body = self.parse_stmt()
            return For(index, cond, update, body)
        if t[0]=="LBRACE":
            return self.parse_block()
        expr = self.parse_expr(); self.eat("SEMI"); return ExprStmt(expr)

    def parse_expr(self, rbp=0):
        """
        This is an implementation of the Pratt parser, which is the 
        most involved part of our transpiler. It takes a lexed epxression
        and returns a parsed expression. ADD MORE WHEN DONE LEARNING ABOUT
        PRATT PARSERS.
     
        Example:
            >>> expr = "4 > a"
            >>> tokens = lex(expr)
            >>> parser = Parser(tokens)
            >>> parsed_expr = parser.parse_expr()
            >>> print(parsed_expr.left.val, parsed_expr.op, parsed_expr.right.name)
            4 > a
        """

        t = self.eat()
        typ,val,_,_ = t
        left = None
        if typ=="NUMBER":
            left = Number(val)
        elif typ=="ID":
            if self.cur()[0]=="LPAREN":
                self.eat("LPAREN")
                args=[]
                if self.cur()[0]!="RPAREN":
                    while True:
                        args.append(self.parse_expr())
                        if self.cur()[0]=="COMMA": self.eat("COMMA"); continue
                        break
                self.eat("RPAREN")
                left = Call(val,args)
            else:
                left = Var(val)
        elif typ=="LPAREN":
            left = self.parse_expr(); self.eat("RPAREN")
        elif typ=="OP" and val in ("-","!"):
            left = Unary(val, self.parse_expr(70))
        else:
            raise SyntaxError(f"Unexpected token in expr: {t}")

        while True:
            cur = self.cur()
            if cur[0]=="OP":
                op = cur[1]
                lbp = infix_bp(op)
                if lbp < rbp: break
                self.eat("OP")
                if is_right_assoc(op):
                    rhs = self.parse_expr(lbp-1)
                else:
                    rhs = self.parse_expr(lbp)
                left = Binary(op, left, rhs)
                continue
            break
        return left

def infix_bp(op):
    """
    Establishes binding power of different opperators to be used in 
    the infix loop for getting the binding power of left operands.

    Argument:
        op (str)
    
    Returns:
        binding power (int)
    """
    if op in ("||",): return 10
    if op in ("&&",): return 20
    if op in ("==","!="): return 30
    if op in ("<",">","<=",">="): return 40
    if op in ("+","-"): return 50
    if op in ("*","/","%"): return 60
    if op == "=": return 5
    return 0


def is_right_assoc(op):
    return False


# Semantic checks

class Sema:
    def __init__(self, prog):
        """
        Takes a Program object and initializes a list of functions to be added by caling Sema.run
        """

        self.prog = prog #instance of a Program object
        self.funcs = {}

    def run(self):
        """
        Fills Sema.funcs, checks for duplicates, assures the existence of a main function, and 
        initiates semantic checking for each function.
        """

        #fill self.funcs. Notice that this is the only time that self.funcs is filled.
        #So, if there is a function in a function, our compiler will raise an error.
        for f in self.prog.funcs:
            if f.name in self.funcs: raise Exception("Duplicate function "+f.name) #checks for duplicates
            self.funcs[f.name]=f
        if "main" not in self.funcs: pass # not required
        for f in self.prog.funcs:
            self.check_func(f) #check each function

    def check_func(self,f):
        """
        Creates a dict of function parameters and calls visit_block on the body of f

        Arguments:
            f (FuncDecl)
        """

        self.locals = {p: True for p in f.params}
        self.visit_block(f.body)
    def visit_block(self, b):
        """
        Vists each statement in the block.

        Arguments:
            b (Block)
        """

        for s in b.stmts:
            self.visit_stmt(s)

    def visit_stmt(self,s):
        """
        Takes a statement and determines its kind. Then, based on this
        information, will visit expressions in this statement.

        Arguments:
            s (VarDecl, Return, If, While, ExprStmt, Block, For)
        """

        if isinstance(s,VarDecl):
            if s.name in self.locals: raise Exception("dup var "+s.name)
            if s.init: self.visit_expr(s.init)
            self.locals[s.name]=True
        elif isinstance(s,Return):
            if s.expr: self.visit_expr(s.expr)
        elif isinstance(s,If):
            self.visit_expr(s.cond); self.visit_stmt(s.then)
            if s.els: self.visit_stmt(s.els)
        elif isinstance(s,While):
            self.visit_expr(s.cond); self.visit_stmt(s.body)
        elif isinstance(s,For):
            self.visit_stmt(s.index); self.visit_expr(s.cond); self.visit_expr(s.update); self.visit_stmt(s.body)
        elif isinstance(s,ExprStmt):
            self.visit_expr(s.expr)
        elif isinstance(s,Block):
            # If s is a block, we want to check its statements as well.
            for st in s.stmts: self.visit_stmt(st)
        else:
            raise Exception("Unknown stmt "+str(s))

    def visit_expr(self,e):
        """
        Takes an expression and, like visit_stmt, determines its kind and performs an
        action based on this.
        
        Arguments:
            e (Number, Var, Call, Unary, Binary) 

        """
        if isinstance(e,Number): return
        if isinstance(e,Var):
            if e.name not in self.locals and e.name not in self.funcs:
                raise Exception("use of undeclared var "+e.name)
            return
        if isinstance(e,Call):
            if e.name not in self.funcs:
                raise Exception("call to unknown "+e.name)
            for a in e.args: self.visit_expr(a)
            return
        if isinstance(e,Unary):
            self.visit_expr(e.expr); return
        if isinstance(e,Binary):
            self.visit_expr(e.left); self.visit_expr(e.right); return
        raise Exception("Unknown expr "+str(e))



# Codegen to Python

class PyGen:
    def __init__(self, prog):
        """
        A PyGen object consists of a AST representation of a c program from which
        it will generate lines of python code.

        """

        self.prog = prog
        #line in kines is a line of code in the resulting python output.
        self.lines = []

    def gen(self):
        """
        Calls gen_func for each function in prog.funcs. Creates a "__main__"
        expression which is appended to lines. Returns string of python output code
        from lines.

        """

        self.lines.append("# generated by c2p")
        for f in self.prog.funcs:
            self.gen_func(f)
        # if there's a main function, call it
        if any(f.name=="main" for f in self.prog.funcs):
            self.lines.append("if __name__=='__main__':")
            self.lines.append("    import sys")
            self.lines.append("    sys.exit(main())")
        return "\n".join(self.lines)

    def gen_func(self,f):
        """
        Takes a function, generates a line of python code declaring the function using
        with the params and name of the fucntion, and then parses the generates the body
        of the function by calling gen_stmt for each statement in the body.
        
        Args:
            f (FuncDecl)
        """

        params = ", ".join(f.params)
        self.lines.append(f"def {f.name}({params}):") #write function decl
        if not f.body.stmts:
            self.lines.append("    pass")
            self.lines.append("")
            return
        for s in f.body.stmts:# append each statement
            self.gen_stmt(s, 1)
        self.lines.append("")

    def gen_stmt(self,s, indent):
        pad = "    "*indent
        if isinstance(s,VarDecl):
            if s.init:
                expr = self.gen_expr(s.init)
                self.lines.append(f"{pad}{s.name} = {expr}")
            else:
                self.lines.append(f"{pad}{s.name} = 0")
        elif isinstance(s,Return):
            if s.expr:
                self.lines.append(f"{pad}return {self.gen_expr(s.expr)}")
            else:
                self.lines.append(f"{pad}return")
        elif isinstance(s,ExprStmt):
            self.lines.append(f"{pad}{self.gen_expr(s.expr)}")
        elif isinstance(s,If):
            self.lines.append(f"{pad}if {self.gen_expr(s.cond)}:")
            self.gen_stmt(s.then, indent+1)
            if s.els:
                self.lines.append(f"{pad}else:")
                self.gen_stmt(s.els, indent+1)
        elif isinstance(s,While):
            self.lines.append(f"{pad}while {self.gen_expr(s.cond)}:")
            self.gen_stmt(s.body, indent+1)
        elif isinstance(s,For):
            #only works for "i<n" conds with i being any index. look up logic in f strings.
            self.lines.append(f"{pad}for {s.index.name} in range({s.cond.right.val}):")
            self.gen_stmt(s.body, indent+1)
        elif isinstance(s,Block):
            for st in s.stmts: self.gen_stmt(st, indent)
        else:
            raise Exception("codegen stmt unknown "+str(s))

    def gen_expr(self,e):
        """
        takes an expression object and returns a string of python code corresponding to that
        expression.

        Args:
            e (ExprStmt)

        Returns:
            python code (str)
            
        """

        if isinstance(e,Number): return str(e.val)
        if isinstance(e,Var): return e.name
        if isinstance(e,Call):
            args = ", ".join(self.gen_expr(a) for a in e.args)
            return f"{e.name}({args})"
        if isinstance(e,Unary):
            if e.op=="!": return f"(not {self.gen_expr(e.expr)})"
            return f"({e.op}{self.gen_expr(e.expr)})"
        if isinstance(e,Binary):
            op = e.op
            if op=="&&": op="and"
            if op=="||": op="or"
            if op=="!": op="not"
            if op =="=":
                return f"{self.gen_expr(e.left)} {op} {self.gen_expr(e.right)}"
            return f"({self.gen_expr(e.left)} {op} {self.gen_expr(e.right)})"

            
        raise Exception("codegen expr unknown "+str(e))



# CLI

def compile_src(src):
    """
    Takes source code, creates tokens, initializes a parser, parses tokens,
    checks program semantics, and generates python code.
    
    Args:
        src (str)

    Returns:
        py (str)

    """

    tokens = lex(src)
    parser = Parser(tokens)
    prog = parser.parse()
    sema = Sema(prog); sema.run()
    py = PyGen(prog).gen()
    return py


def main_cli():
    """
    CLI USAGE: python c2p.py src.c > out.py
    """

    if len(sys.argv) < 2:
        sys.exit(1)
    src = open(sys.argv[1]).read()
    py = compile_src(src)
    print(py)


if __name__=="__main__":
    main_cli()
