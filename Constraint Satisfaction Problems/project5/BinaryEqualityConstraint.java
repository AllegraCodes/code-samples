package project5;

import java.util.ArrayList;

public class BinaryEqualityConstraint extends AbstractBinaryConstraint implements Constraint {
    boolean equal; // true = the 2 items must be in the same bag, false = the 2 items must be in different bags

    public BinaryEqualityConstraint(Item item1, Item item2, boolean equal) {
        super(item1, item2);
        this.equal = equal;
    }

    public boolean check(ArrayList<Item> items, ArrayList<Bag> bags) {
        // both items must have assignments to violate this constraint
        if (!item1.isAssigned || !item2.isAssigned) return true;

        boolean sameBag = false;
        for (Bag b : bags) {
            if (b.contents.contains(item1) && b.contents.contains(item2)) {
                sameBag = true;
            }
        }

        if (equal) {
            return sameBag;
        } else {
            return !sameBag;
        }
    }
}
