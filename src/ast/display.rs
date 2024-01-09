use std::fmt::{Debug, Display, Write};

use indenter::indented;

use super::*;

impl Display for CompUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.func_def)
    }
}

impl Display for FuncDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}() {}", self.func_type, self.ident, self.block)
    }
}

impl Display for FuncType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FuncType::Int => write!(f, "int"),
        }
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for item in &self.items {
            let mut f = indented(f).with_str("    ");
            write!(f, "{}", item)?;
        }
        write!(f, "}}")
    }
}

impl Display for BlockItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockItem::Decl { decl } => write!(f, "{}", decl),
            BlockItem::Stmt { stmt } => write!(f, "{}", stmt),
        }
    }
}

impl Display for Decl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decl::Const(decl) => write!(f, "{}", decl),
            Decl::Var(decl) => write!(f, "{}", decl),
        }
    }
}

impl Display for ConstDecl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "const {}", self.ty)?;
        for (i, def) in self.defs.iter().enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            write!(f, " {}", def)?;
        }
        writeln!(f, ";")
    }
}

impl Display for VarDecl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ty)?;
        for (i, def) in self.defs.iter().enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            write!(f, " {}", def)?;
        }
        writeln!(f, ";")
    }
}

impl Display for BType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BType::Int => write!(f, "int"),
        }
    }
}

impl Display for ConstDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}", self.ident, self.expr)
    }
}

impl Display for ConstExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr)
    }
}

impl Display for VarDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ident)?;
        if let Some(init) = &self.init {
            write!(f, " = {}", init)?;
        }
        Ok(())
    }
}

impl Display for InitVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr)
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stmt::Assign { ident, expr } => {
                writeln!(f, "{} = {};", ident, expr)
            }
            Stmt::Expr { expr } => {
                if let Some(expr) = expr {
                    writeln!(f, "{};", expr)
                } else {
                    writeln!(f, ";")
                }
            }
            Stmt::Block { block } => {
                writeln!(f, "{}", block)
            }
            Stmt::If { cond, then, els } => {
                write!(f, "if ({}) ", cond)?;
                write!(f, "{}", then)?;
                if let Some(els) = els {
                    write!(f, "else {}", els)?;
                }
                Ok(())
            }
            Stmt::While { cond, body } => {
                write!(f, "while ({})", cond)?;
                write!(f, "{}", body)
            }
            Stmt::Break => {
                writeln!(f, "break;")
            }
            Stmt::Continue => {
                writeln!(f, "continue;")
            }
            Stmt::Return { expr } => {
                writeln!(f, "return {};", expr)
            }
        }
    }
}

impl Expr {
    fn precedence(&self) -> u32 {
        match self {
            Expr::Number(_) | Expr::LVar(_) => 0,
            Expr::Unary { .. } => 1,
            Expr::Binary { op, .. } => match op.node {
                BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
                BinaryOp::Add | BinaryOp::Sub => 3,
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
                BinaryOp::Eq | BinaryOp::Ne => 5,
                BinaryOp::LAnd => 6,
                BinaryOp::LOr => 7,
            },
        }
    }

    fn fmt_with_prec(&self, f: &mut std::fmt::Formatter<'_>, outer_prec: u32) -> std::fmt::Result {
        let inner_prec = self.precedence();
        if inner_prec > outer_prec {
            write!(f, "(")?;
        }
        match self {
            Expr::Number(expr) => write!(f, "{}", expr),
            Expr::LVar(ident) => write!(f, "{}", ident),
            Expr::Unary { op, expr } => {
                write!(f, "{}", op)?;
                match op.node {
                    UnaryOp::Pos | UnaryOp::Neg => expr.fmt_with_prec(f, inner_prec + 1)?,
                    UnaryOp::Not => expr.fmt_with_prec(f, inner_prec)?,
                }

                Ok(())
            }
            Expr::Binary { op, lhs, rhs } => {
                lhs.fmt_with_prec(f, inner_prec)?;
                write!(f, " {} ", op)?;
                rhs.fmt_with_prec(f, inner_prec)?;
                Ok(())
            }
        }?;
        if inner_prec > outer_prec {
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with_prec(f, 15)
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Pos => write!(f, "+"),
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Ne => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::LAnd => write!(f, "&&"),
            BinaryOp::LOr => write!(f, "||"),
        }
    }
}

pub struct Displayable<T>(pub T);

impl<T: Display> Debug for Displayable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
