import { useState, useEffect } from "react";
import "./style.less";

const Comment = ({username, comment, spoiler}) => {
    const [isSpoiler, setSpoiler] = useState(spoiler);

    useEffect(() => {
        setSpoiler(spoiler);
    }, [username, comment, spoiler]);

    return (
        <div className="comment">
            <p>{username}</p>
            <h2 className={`content ${isSpoiler ? "spoiler" : ""}`} onClick={() => setSpoiler(false)}>{comment}</h2>
        </div>
    );
};

export default Comment;