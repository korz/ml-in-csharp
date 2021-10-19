//<Query Kind="Program" />

void Main()
{
    var p = true;
    var q = true;

    Xor(true,true).Dump();
    Xor(true,false).Dump();
    Xor(false,true).Dump();
    Xor(false,false).Dump();
}

public bool IsMutuallyExclusive(bool p, bool q)
{
    if (p && !q || q && !p)
    {
        return true;
    }

    return false;
}
