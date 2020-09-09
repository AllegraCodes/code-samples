package project5;

import jdk.internal.util.xml.impl.Input;

import java.util.ArrayList;

public class Item {
    String name; // The name of this item as a single upper case letter
    int weight; // The weight of this item in kg
    boolean isAssigned; // true if this item has been assigned
    ArrayList<Constraint> constraints; // the constraints involving this Item
    ArrayList<Bag> domain;

    public Item(String name, int weight){
        this.name = name;
        this.weight = weight;
        this.isAssigned = false;
        this.constraints = new ArrayList<>();
        this.domain = new ArrayList<>();
    }

    public Item(String name, String weight){
        this.name = name;
        this.weight = Integer.parseInt(weight);
        this.isAssigned = false;
        this.constraints = new ArrayList<>();
        this.domain = new ArrayList<>();
    }

    public void addConstraint(Constraint c) {
        this.constraints.add(c);
    }

    public void addValue(Bag b) {
        this.domain.add(b);
    }

    public void removeValue(Bag b) {
        this.domain.remove(b);
    }

    public String getName() {
        return this.name;
    }

}
