package project5;

public abstract class AbstractBinaryConstraint {
    Item item1, item2; // the 2 items this constraint applies to

    public AbstractBinaryConstraint(Item item1, Item item2) {
        this.item1 = item1;
        this.item2 = item2;
    }
}
