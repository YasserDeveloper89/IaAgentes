def format_currency(value, currency="S/"):
    """
    Formatea número como moneda local, con símbolo y dos decimales.
    """
    try:
        return f"{currency} {value:,.2f}"
    except:
        return value
