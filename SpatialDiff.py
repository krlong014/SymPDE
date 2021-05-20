
def spatialDiff(expr, op):

    # Make sure arguments are valid
    assert(isinstance(expr, Expr) and isinstance(op, HungryDiffOp))
    assert(not isinstance(expr, AggExpr))

    if expr.isSpatialConstant():
        return 0

    if isinstance(expr, SumExpr):
        return spatialDiff(expr.L, op) + spatialDiff(expr.R, op)

    if isinstance(expr, UnaryMinus):
        return -spatialDiff(expr, op)

    if isinstance(expr, ProductExpr):
        return differentiateProduct(expr, op)

    if isinstance(expr, CrossProductExpr):
        return differentiateCrossProduct(expr, op)

    if isinstance(expr, PowerExpr):
        return differentiatePower(expr, op)

    if isinstance(expr, QuotientExpr):
        return differentiateQuotient(expr, op)

    if isinstance(expr, CoordinateExpr):
        return differentiateCoordinate(expr, op)

    if isinstance(expr, VectorExprInterface):
        return differentiateCoordinate(expr, op)


# def differentiateProduct(expr, op):
#     assert(isinstance(expr, ProductExpr))
#
#     if
